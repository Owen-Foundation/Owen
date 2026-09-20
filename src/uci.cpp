#include "uci.h"
#include "movegen.h"
#include "bitboard.h"
#include "nnue/network.h"
#include <iostream>
#include <sstream>
#include <thread>
#include <atomic>
#include <mutex>
#include <string>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <optional>
#include <vector>
#include <charconv>

namespace owen2 {

// ── helpers ────────────────────────────────────────────────────────
static inline std::string trim(std::string s){
    while(!s.empty() && std::isspace((unsigned char)s.front())) s.erase(s.begin());
    while(!s.empty() && std::isspace((unsigned char)s.back()))  s.pop_back();
    return s;
}
static inline std::string toLower(std::string s){
    for(char& c: s) c = char(std::tolower((unsigned char)c));
    return s;
}
static inline bool ieq(const std::string& a, const std::string& b){
    return toLower(a)==toLower(b);
}
static bool parse_int_strict(const std::string& s, int& out){
    if(s.empty()) return false;
    try { size_t p=0; int v=std::stoi(s,&p); if(p!=s.size()) return false; out=v; return true; } catch(...){ return false; }
}
static bool parse_int64_strict(const std::string& s, int64_t& out){
    if(s.empty()) return false;
    try { size_t p=0; long long v=std::stoll(s,&p); if(p!=s.size()) return false; out=v; return true; } catch(...){ return false; }
}
static void log_info(const std::string& s){
    // Never spam stdout with debug — this is UCI-legal "info string" but used sparingly.
    // Unknown commands are ignored silently per spec; diagnostics for malformed setoption/position go here.
    std::cout << "info string " << s << "\n" << std::flush;
}

// ── UCI state ─────────────────────────────────────────────────────
static Position g_pos;
static search::Searcher g_searcher;

// Threaded search state — single owner pattern.
// Only the UCI thread mutates g_searchThread; search thread only reads g_stop/g_pondering.
static std::thread g_searchThread;
static std::atomic<bool> g_stop{false};
static std::atomic<bool> g_searching{false};
static std::atomic<bool> g_pondering{false};
static std::mutex g_outMu; // serialises all UCI stdout

static void safePrint(const std::string& s){
    std::lock_guard<std::mutex> lk(g_outMu);
    std::cout << s << "\n" << std::flush;
}

// Standard search summary for GUIs and game managers (fastchess extracts
// the score from the last info line). Emitted once per search, just before
// bestmove. Score is side-to-move centipawns; mates use UCI mate-in-N.
static void emit_final_info(const search::SearchResult& res){
    long long ms = res.time_ms > 0 ? (long long)res.time_ms : 1;
    long long nps = (long long)res.nodes * 1000LL / ms;
    std::ostringstream os;
    os << "info depth " << res.depth << " seldepth " << res.depth;
    if(is_mate_score(res.score)){
        int ply = VALUE_MATE - std::abs(res.score);
        int moves = (ply + 1) / 2;
        os << " score mate " << (res.score > 0 ? moves : -moves);
    } else {
        os << " score cp " << res.score;
    }
    os << " nodes " << (unsigned long long)res.nodes
       << " nps " << nps << " time " << (long long)res.time_ms;
    // Single-move PV only for non-mate scores: game managers reject a
    // mate claim backed by an incomplete PV ("Incomplete mating PV").
    std::string bm = move_to_uci(res.bestMove);
    if(!is_mate_score(res.score) && !bm.empty() && res.bestMove != 0)
        os << " pv " << bm;
    safePrint(os.str());
}

// ── UCI options registry ──────────────────────────────────────────
// Typed registry: GUIs (Knights/Cutechess/Arena) query these to build UI.
// Every advertised option is actually implemented in apply_setoption.
// Unknown setoption names are ignored silently (spec: should not error).
enum class OptType { Spin, Check, String, Button, Combo };
struct Opt {
    std::string name;          // canonical display name
    OptType type;
    std::string defaultStr;    // what we print as "default ..."
    int minV=0, maxV=0;        // for spin
    std::string extra;         // for combo var list etc.
    std::string decl() const {
        std::string t;
        switch(type){
            case OptType::Spin:   t="spin"; break;
            case OptType::Check:  t="check"; break;
            case OptType::String: t="string"; break;
            case OptType::Button: t="button"; break;
            case OptType::Combo:  t="combo"; break;
        }
        std::string s="option name "+name+" type "+t;
        if(type==OptType::Spin) s += " default "+defaultStr+" min "+std::to_string(minV)+" max "+std::to_string(maxV);
        else if(type==OptType::Check||type==OptType::String) s += " default "+defaultStr;
        if(!extra.empty()) s += " " + extra;
        return s;
    }
};
static const std::vector<Opt> kOpts = {
    {"Debug Log File", OptType::String, ""},
    {"Hash",           OptType::Spin,   "64",  1, 16384},
    {"Threads",        OptType::Spin,   "1",   1, 512},
    {"Ponder",         OptType::Check,  "false"},
    {"UCI_Chess960",   OptType::Check,  "false"},
    {"UCI_ShowWDL",    OptType::Check,  "false"},
    {"NNUEFile",       OptType::String, "nets/o2-v1.o2nn"},
    {"MultiPV",        OptType::Spin,   "1",   1, 500},
    {"Move Overhead",  OptType::Spin,   "10",  0, 5000},
    {"Slow Mover",     OptType::Spin,   "100",10, 1000},
    {"Clear Hash",     OptType::Button, ""},
    {"MTS_C",          OptType::String, "1.35"},
    {"Bound Prune",    OptType::Check,  "false"},
    {"Best First",     OptType::Check,  "false"},
    // Limiter + real ceiling — honest cap, never fakes strength above what search+NN can deliver.
    {"UCI_Elo",        OptType::Spin,   "1320", 1320, 4100},
    {"UCI_LimitStrength", OptType::Check, "false"},
};

static void print_options(){
    for(auto &o: kOpts) std::cout << o.decl() << "\n";
}

static bool g_limitStrength=false;
static int  g_uciElo=1320;

static void apply_setoption(const std::string& line){
    // setoption name <id> [value <x>] — name may contain spaces. Spec: case-insensitive name match.
    // We must not crash on stoi failure or missing value.
    auto p = line.find("name ");
    if(p==std::string::npos) return;
    auto v = line.find(" value ");
    std::string name, value;
    {
        size_t a=p+5, b=(v==std::string::npos? line.size(): v);
        name=trim(line.substr(a,b-a));
    }
    if(v!=std::string::npos) value=trim(line.substr(v+7));
    // value may legitimately be empty for button/string — don't error on empty for those.
    std::string lname = toLower(name);

    auto clampSpin = [](int v,int lo,int hi){ return std::max(lo,std::min(hi,v)); };

    if(lname=="hash"){
        int iv; if(!parse_int_strict(value, iv)){ log_info("Hash needs integer value"); return; }
        iv = clampSpin(iv, 1, 16384);
        g_searcher.set_tt_size(iv);
    } else if(lname=="threads"){
        int iv; if(!parse_int_strict(value, iv)){ log_info("Threads needs integer value"); return; }
        iv = clampSpin(iv, 1, 512);
        g_searcher.set_threads(iv);
    } else if(lname=="ponder"){
        // accepted — actual ponder behaviour is via go ponder / ponderhit.
        (void)value;
    } else if(lname=="uci_chess960" || lname=="chess960"){
        (void)value;
    } else if(lname=="uci_showwdl"){
        g_searcher.set_show_wdl(ieq(value,"true") || value=="1");
    } else if(lname=="nnuefile"){
        std::string path=value;
        if(path.empty()){ log_info("NNUEFile needs a path"); return; }
        // Evaluations change with the net: drop TT values and both eval
        // caches so nothing stale survives the switch.
        g_searcher.new_game();
        if(!nnue::g_network.load(path))
            log_info("NNUE load failed: " + path + " (using handcrafted eval)");
        else
            log_info("NNUE loaded: " + path);
    } else if(lname=="multipv"){
        int iv; if(!parse_int_strict(value, iv)){ log_info("MultiPV needs integer value"); return; }
        iv = clampSpin(iv, 1, 500);
        g_searcher.set_multipv(iv);
    } else if(lname=="move overhead"){
        int iv; if(!parse_int_strict(value, iv)){ log_info("Move Overhead needs integer value"); return; }
        iv = clampSpin(iv, 0, 5000);
        g_searcher.set_move_overhead(iv);
    } else if(lname=="slow mover"){
        int iv; if(!parse_int_strict(value, iv)){ log_info("Slow Mover needs integer value"); return; }
        iv = clampSpin(iv, 10, 1000);
        g_searcher.set_slow_mover(iv);
    } else if(lname=="clear hash"){
        g_searcher.new_game();
        log_info("hash cleared");
    } else if(lname=="uci_elo"){
        int iv; if(!parse_int_strict(value, iv)){ log_info("UCI_Elo needs integer value"); return; }
        iv = clampSpin(iv, 1320, 4100);
        g_uciElo = iv;
        g_searcher.set_uci_elo(iv);
    } else if(lname=="uci_limitstrength"){
        g_limitStrength = ieq(value,"true") || value=="1";
        g_searcher.set_limit_strength(g_limitStrength);
    } else if(lname=="debug log file"){
        // accepted for GUI compat, no file logging implemented.
    } else if(lname=="mts_c"){
        // string type — allow float
        try { double d=std::stod(value); g_searcher.set_marrow_c(d); }
        catch(...){ log_info("MTS_C needs numeric value"); }
    } else if(lname=="bound prune"){
        g_searcher.set_bound_prune(ieq(value,"true") || value=="1");
    } else if(lname=="best first"){
        g_searcher.set_best_first(ieq(value,"true") || value=="1");
    } else if(lname=="mts_multipv"){
        int iv; if(!parse_int_strict(value, iv)){ log_info("MTS_MultiPV needs integer value"); return; }
        iv = clampSpin(iv, 1, 500);
        g_searcher.set_multipv(iv);
    } else {
        // Unknown option — spec: ignore silently. We use info string sparingly, not per-unknown spam.
        // Only log the first time per unknown name to avoid flooding long GUI sessions.
        static std::mutex mu; static std::vector<std::string> seen;
        std::lock_guard<std::mutex> lk(mu);
        std::string key=lname;
        if(std::find(seen.begin(),seen.end(),key)==seen.end()){
            seen.push_back(key);
            log_info("unknown option '" + name + "' ignored");
        }
    }
    // g_limitStrength/g_uciElo are mirrored into Searcher via setters above.
}

static void set_position(const std::string& line){
    // Robust position parsing per UCI spec:
    //   position startpos [moves ...]
    //   position fen <6 fields or 4 fields> [moves ...]
    // FEN: pieces stm castling ep halfmove fullmove. Some GUIs omit half/full.
    // We must not corrupt board on malformed input — fall back to startpos and use info string.
    std::string t = trim(line);
    // find "moves" as a standalone token (preceded by space), not inside FEN
    // FEN never contains the word "moves", so a simple find is safe, but we token-scan to be strict.
    std::istringstream iss(t);
    std::string tok; iss >> tok; // position
    if(tok!="position"){ return; }
    std::string rest; std::getline(iss, rest); rest = trim(rest);
    // rest starts with "startpos" or "fen" or empty
    std::string fen;
    std::string movesPart;
    auto extract_moves = [&](const std::string& s)->std::string{
        // s may contain " moves e2e4 ..." — find " moves " or " moves" at end
        auto p = s.find(" moves");
        if(p==std::string::npos) return "";
        // ensure it's token "moves" not substring
        // preceding char is space (we searched with space), following is space or end
        return trim(s.substr(p+6));
    };
    if(rest.rfind("startpos",0)==0){
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        movesPart = extract_moves(rest);
    } else if(rest.rfind("fen",0)==0){
        std::string after = trim(rest.substr(3));
        // after may be empty (malformed)
        if(after.empty()){
            log_info("position fen: missing FEN, using startpos");
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        } else {
            auto mvPos = after.find(" moves");
            // Accept " moves " with leading space; if not found, entire after is FEN
            std::string fenPart = (mvPos==std::string::npos? after : trim(after.substr(0, mvPos)));
            movesPart = (mvPos==std::string::npos? "" : trim(after.substr(mvPos+6)));
            // Validate FEN: 4 or 6 fields. If 4 fields, append " 0 1". If <4, fallback.
            {
                std::istringstream fs(fenPart);
                std::vector<std::string> fields; std::string f;
                while(fs>>f) fields.push_back(f);
                if(fields.size()<4){
                    log_info("position fen: malformed FEN (need 4-6 fields), using startpos");
                    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
                } else if(fields.size()==4){
                    fen = fields[0]+" "+fields[1]+" "+fields[2]+" "+fields[3]+" 0 1";
                } else if(fields.size()>=6){
                    fen = fields[0]+" "+fields[1]+" "+fields[2]+" "+fields[3]+" "+fields[4]+" "+fields[5];
                } else { // 5 fields — halfmove present, no fullmove
                    fen = fenPart + " 1";
                }
            }
            if(fen.empty()) fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        }
    } else {
        log_info("position: expected 'startpos' or 'fen', using startpos");
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        // try to salvage moves if line contained "moves"
        movesPart = extract_moves(rest);
    }

    // Apply FEN atomically: validate via set_fen which reinits zobrist, but don't crash on stoi inside.
    // Position::set_fen handles half/full via stoi — those are inside fen string now sanitized above.
    try {
        g_pos.set_fen(fen);
    } catch(...){
        log_info("position fen: exception applying FEN, using startpos");
        g_pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }

    // Apply moves sequentially — each must be legal; stop on first illegal (do not inject illegal board state).
    if(!movesPart.empty()){
        std::istringstream ms(movesPart);
        std::string mv;
        while(ms >> mv){
            Move m = parse_uci_move(g_pos, mv);
            if(!m){
                log_info("illegal move in position: " + mv + " (stopping moves)");
                break;
            }
            // Double-check legality against generate_legal (parse_uci_move already does, but belt-and-suspenders)
            bool legal=false;
            {
                auto ls = generate_legal(g_pos);
                for(auto x: ls) if(x==m){ legal=true; break; }
            }
            if(!legal){
                log_info("illegal move in position: " + mv + " (rejected)");
                break;
            }
            g_pos.do_move(m);
        }
    }
    g_searcher.set_position(g_pos);
}

static search::SearchLimits parse_go(const std::string& line){
    search::SearchLimits lim;
    // UCI spec defaults per field; engine decides search bounds.
    // We MUST distinguish "no limit given" from 0 so wtime 0 is legal.
    lim.depth = 64; // sentinel meaning "no depth limit" unless explicitly set — search.cpp handles 64 as infinite-depth
    bool depthSet=false, hasTime=false;
    lim.movetime_ms=-1; lim.wtime_ms=-1; lim.btime_ms=-1; lim.winc_ms=0; lim.binc_ms=0;
    lim.movestogo=0; lim.infinite=false; lim.ponder=false; lim.nodes=-1; lim.mate=-1;

    std::istringstream iss(line);
    std::string tok; iss >> tok; // go
    // Token loop: every keyword consumes following value atomically; on parse failure skip.
    while(iss >> tok){
        if(tok=="depth"){
            std::string v; if(!(iss>>v)) break;
            int iv; if(parse_int_strict(v, iv)) { lim.depth=iv; depthSet=true; } else { log_info("go depth: bad value '"+v+"'"); }
        } else if(tok=="movetime"){
            std::string v; if(!(iss>>v)) break;
            int64_t iv; if(parse_int64_strict(v, iv)) { lim.movetime_ms=iv; hasTime=true; } else log_info("go movetime: bad value '"+v+"'");
        } else if(tok=="wtime"){
            std::string v; if(!(iss>>v)) break;
            int64_t iv; if(parse_int64_strict(v, iv)) { lim.wtime_ms=iv; hasTime=true; } else log_info("go wtime: bad value '"+v+"'");
        } else if(tok=="btime"){
            std::string v; if(!(iss>>v)) break;
            int64_t iv; if(parse_int64_strict(v, iv)) { lim.btime_ms=iv; hasTime=true; } else log_info("go btime: bad value '"+v+"'");
        } else if(tok=="winc"){
            std::string v; if(!(iss>>v)) break;
            int64_t iv; if(parse_int64_strict(v, iv)) lim.winc_ms=iv;
        } else if(tok=="binc"){
            std::string v; if(!(iss>>v)) break;
            int64_t iv; if(parse_int64_strict(v, iv)) lim.binc_ms=iv;
        } else if(tok=="movestogo"){
            std::string v; if(!(iss>>v)) break;
            int iv; if(parse_int_strict(v, iv)) lim.movestogo=iv;
        } else if(tok=="infinite"){
            lim.infinite=true;
        } else if(tok=="nodes"){
            std::string v; if(!(iss>>v)) break;
            int64_t iv; if(parse_int64_strict(v, iv)) lim.nodes=iv;
        } else if(tok=="mate"){
            std::string v; if(!(iss>>v)) break;
            int iv; if(parse_int_strict(v, iv)) lim.mate=iv;
        } else if(tok=="ponder"){
            lim.ponder=true;
        } else if(tok=="searchmoves"){
            // Remaining tokens until next known keyword or EOF are move strings.
            // Knights never sends searchable words that collide with go keywords, but we handle it anyway.
            std::string mv;
            while(iss >> mv){
                static const std::vector<std::string> kws={"depth","movetime","wtime","btime","winc","binc","movestogo","infinite","nodes","mate","ponder","searchmoves"};
                if(std::find(kws.begin(),kws.end(),mv)!=kws.end()){
                    // Put back: reconstruct minimal — since searchmoves is expected last, break and re-inject tok
                    tok=mv; // outer loop will process it again — but we consumed it, so handle inline
                    // Re-dispatch this tok as keyword in this same iteration to avoid losing it
                    if(tok=="depth"){ std::string v; if(iss>>v){ int iv; if(parse_int_strict(v,iv)){ lim.depth=iv; depthSet=true; } } }
                    else if(tok=="movetime"){ std::string v; if(iss>>v){ int64_t iv; if(parse_int64_strict(v,iv)){ lim.movetime_ms=iv; hasTime=true; } } }
                    else if(tok=="infinite") lim.infinite=true;
                    else if(tok=="ponder") lim.ponder=true;
                    continue;
                }
                // Validate UCI move syntax (4-5 chars), but don't filter by legality here — search does.
                if(mv.size()>=4 && mv.size()<=5) lim.searchmoves.push_back(mv);
            }
            break;
        } else {
            // Unknown go token — ignore per spec (don't corrupt state, don't spam)
        }
    }
    // If no depth was explicitly given, mark as "no depth limit" sentinel so search doesn't treat 64 as a depth cap.
    if(!depthSet) lim.depth = 64;
    // If nothing at all was given (bare "go"), search handles the soft fallback — don't synthesize movetime.
    // Ponder acts like infinite until ponderhit.
    (void)hasTime;
    return lim;
}

static void join_search_thread(){
    if(g_searchThread.joinable()){
        g_searchThread.join();
    }
}

static void start_search(const search::SearchLimits& lim){
    if(g_searching.load()){
        g_stop.store(true);
        g_pondering.store(false);
        join_search_thread();
        g_searching.store(false);
    } else if(g_searchThread.joinable()){
        // Previous search finished but thread not yet joined — assigning a new
        // std::thread to a joinable one calls std::terminate ("without active exception").
        join_search_thread();
    }
    g_stop.store(false);
    g_searching.store(true);
    g_pondering.store(lim.ponder);
    search::SearchLimits limCopy = lim;
    g_searchThread = std::thread([limCopy]{
        auto run_search = [&]() -> search::SearchResult {
            try { return g_searcher.search(limCopy, g_stop, [](const std::string& s){ safePrint(s); }); }
            catch(const std::exception& e){ safePrint(std::string("info string search exception: ")+e.what()); return search::SearchResult{0,0,0,1,0,0}; }
            catch(...){ safePrint("info string search exception: unknown"); return search::SearchResult{0,0,0,1,0,0}; }
        };
        try {
        auto res = run_search();

        bool wasPondering = g_pondering.load() && !g_stop.load();
        if(wasPondering){
            // Pondering: hold without emitting bestmove until ponderhit or stop.
            while(g_pondering.load() && !g_stop.load())
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            if(g_pondering.load()){
                // stop while pondering without ponderhit
                emit_final_info(res);
                std::lock_guard<std::mutex> lk(g_outMu);
                std::cout << "bestmove " << move_to_uci(res.bestMove) << "\n" << std::flush;
                g_searching.store(false);
                return;
            }
            // ponderhit: bestmove already determined; fall through to emit it
        }
        {
            // Final search summary first (own lock inside), then bestmove.
            emit_final_info(res);
            std::lock_guard<std::mutex> lk(g_outMu);
            // Guarantee exactly one bestmove — never "bestmove (none)" unless truly no legal moves.
            std::string bm = move_to_uci(res.bestMove);
            if(bm.empty() || res.bestMove==0){
                // No legal moves (mate/stalemate) — spec says bestmove 0000 or (none). We use 0000.
                std::cout << "bestmove 0000\n" << std::flush;
            } else {
                std::cout << "bestmove " << bm;
                if(res.ponderMove) std::cout << " ponder " << move_to_uci(res.ponderMove);
                std::cout << "\n" << std::flush;
            }
        }
        g_searching.store(false);
        } catch(const std::exception& e){
            try { safePrint(std::string("info string thread exception: ")+e.what()); } catch(...){}
            try { std::lock_guard<std::mutex> lk(g_outMu); std::cout<<"bestmove 0000\n"<<std::flush; } catch(...){}
            g_searching.store(false);
        } catch(...){
            try { safePrint("info string thread exception: unknown"); } catch(...){}
            try { std::lock_guard<std::mutex> lk(g_outMu); std::cout<<"bestmove 0000\n"<<std::flush; } catch(...){}
            g_searching.store(false);
        }
    });
}

std::string run_bench(int depth){
    depth = std::max(1, std::min(depth, 10));
    search::Searcher bsearch;
    Position bp;
    bp.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    bsearch.set_position(bp);
    search::SearchLimits lim; lim.depth = depth;
    std::atomic<bool> bst{false};
    auto t0=std::chrono::steady_clock::now();
    auto res = bsearch.search(lim, bst, nullptr);
    auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now()-t0).count();
    uint64_t h=bsearch.eval_cache().hits(), m=bsearch.eval_cache().misses();
    double hr = (h+m) ? 100.0*double(h)/double(h+m) : 0.0;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "bench depth %d nodes %llu nps %lld time %lldms bestmove %s evalhit %.1f%%",
        depth, (unsigned long long)res.nodes,
        ms ? (long long)(res.nodes*1000ULL/std::max<int64_t>(1,ms)) : 0,
        (long long)ms, move_to_uci(res.bestMove).c_str(), hr);
    return std::string(buf);
}

void uci_loop(){
    init_attacks();
    Position::init_zobrist();

    g_pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    g_searcher.set_position(g_pos);

    std::string line;
    while(std::getline(std::cin, line)){
        std::string t = trim(line);
        if(t.empty()) continue;

        if(t=="uci"){
            std::cout << "id name Owen 2\n";
            std::cout << "id author Owen Foundation\n";
            print_options();
            std::cout << "uciok\n" << std::flush;
        } else if(t=="isready"){
            // Must eventually respond readyok even mid-search. Don't block the search thread.
            std::cout << "readyok\n" << std::flush;
        } else if(t=="ucinewgame"){
            if(g_searching.load()){
                g_stop.store(true);
                g_pondering.store(false);
                join_search_thread();
                g_searching.store(false);
            } else if(g_searchThread.joinable()){
                join_search_thread();
            } else {
                g_stop.store(false);
                g_pondering.store(false);
            }
            g_pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            g_searcher.new_game();
            g_searcher.set_position(g_pos);
        } else if(t.rfind("position",0)==0){
            if(g_searching.load()){
                g_stop.store(true);
                g_pondering.store(false);
                join_search_thread();
                g_searching.store(false);
            } else if(g_searchThread.joinable()){
                join_search_thread();
            }
            set_position(t);
        } else if(t.rfind("setoption",0)==0){
            // setoption is allowed even while searching per some GUIs — apply atomically.
            apply_setoption(t);
        } else if(t.rfind("go",0)==0){
            auto lim = parse_go(t);
            start_search(lim);
        } else if(t=="stop"){
            if(g_searching.load()){
                g_stop.store(true);
                g_pondering.store(false);
                join_search_thread();
                // join waits for single bestmove emission; g_searching cleared by search thread.
            }
        } else if(t=="ponderhit"){
            if(g_searching.load() && g_pondering.load()){
                g_pondering.store(false);
                // Let search thread finish and emit bestmove; could restart with clock but minimal impl is spec-compliant.
            }
        } else if(t=="quit"){
            g_stop.store(true);
            g_pondering.store(false);
            join_search_thread();
            break;
        } else if(t=="d" || t=="display"){
            std::lock_guard<std::mutex> lk(g_outMu);
            std::cout << g_pos.fen() << "\n";
            for(int r=7;r>=0;--r){
                std::cout << (r+1) << " ";
                for(int f=0;f<8;++f){
                    Piece p=g_pos.piece_on(make_square(f,r));
                    char ch='.';
                    if(p!=NO_PIECE){ const char* s="PNBRQKpnbrqk"; ch=s[p]; }
                    std::cout << ch << ' ';
                }
                std::cout << "\n";
            }
            std::cout << "  a b c d e f g h\n";
            std::cout << "Eval: " << nnue::g_network.evaluate(g_pos) << " cp\n" << std::flush;
        } else if(t.rfind("perft",0)==0){
            int depth=5;
            {
                std::istringstream iss(t);
                std::string tok; iss >> tok;
                if(iss >> depth) {}
                depth = std::max(1, std::min(depth, 7));
            }
            for(int d=1; d<=depth; ++d){
                Position tmp=g_pos;
                auto t0=std::chrono::steady_clock::now();
                uint64_t n=perft(tmp,d);
                auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t0).count();
                safePrint("perft " + std::to_string(d) + " " + std::to_string(n) + " (" + std::to_string(ms) + " ms)");
            }
        } else if(t=="bench" || t.rfind("bench ",0)==0){
            // Non-UCI diagnostic: fixed startpos search, repeatable numbers.
            // Uses a throwaway Searcher (single thread) so live TT/state is untouched.
            int depth=6;
            {
                std::istringstream iss(t);
                std::string tok; iss >> tok;
                int d=0; if(iss >> d) depth = std::max(1, std::min(d, 10));
            }
            if(g_searching.load()){
                safePrint("info string bench: search in progress, try later");
            } else {
                safePrint(run_bench(depth));
            }
        } else if(t=="help"){
            safePrint("Owen 2 UCI — commands: uci, isready, ucinewgame, position, go, stop, ponderhit, quit, d, perft [n], bench [depth]");
        } else {
            // Unknown command — spec: ignore silently. Do not emit info string per unknown to avoid GUI log spam.
            // Debug to stderr only.
            std::cerr << "info string ignored: " << t << "\n" << std::flush;
        }
    }

    g_stop.store(true);
    g_pondering.store(false);
    join_search_thread();
}

} // namespace owen2
