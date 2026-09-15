#include "position.h"
#include "movegen.h"
#include "bitboard.h"
#include "search/search.h"
#include "nnue/network.h"
#include "binpack.h"
#include <iostream>
#include <fstream>
#include <random>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <chrono>
#include <memory>
#include <filesystem>

#pragma pack(push,1)
struct SDataRecord {
    uint8_t board[64];
    uint8_t stm;
    int16_t eval;
    uint8_t result;
    uint8_t ply;
    uint8_t castling; // bit0 K, bit1 Q, bit2 k, bit3 q
    uint8_t ep;       // 0..63 square, 64 = none
    uint16_t move16;  // v3 policy label: from6|to6|promo3(0 none,1 N,2 B,3 R,4 Q)|ep1
};
#pragma pack(pop)
static_assert(sizeof(SDataRecord)==73, "pack broken");

static void play_games(int thread_id, int games, int movetime_ms, int depth,
                       const std::string& net_path, const std::string& out_path,
                       const std::string& binpack_path,
                       const std::vector<std::string>* book,
                       int resign_cp, int resign_moves,
                       std::atomic<int>& done, std::atomic<uint64_t>& positions,
                       std::mutex& io_mtx)
{
    owen2::search::Searcher searcher;
    searcher.set_tt_size(4); // small per-thread
    if(!net_path.empty()){
        std::lock_guard<std::mutex> lk(io_mtx);
        if(owen2::nnue::g_network.load(net_path))
            std::cout << "[t" << thread_id << "] loaded net " << net_path << "\n";
    }

    std::mt19937 rng(42 + thread_id * 1009);
    std::string part = out_path + ".part" + std::to_string(thread_id);
    std::ofstream f(part, std::ios::binary);
    if(!f){ std::cerr << "cannot open " << part << "\n"; return; }
    std::unique_ptr<binpack::Writer> bpw;
    if(!binpack_path.empty()){
        std::string bppart = binpack_path + ".part" + std::to_string(thread_id);
        try { bpw = std::make_unique<binpack::Writer>(bppart); }
        catch(const std::exception& e){ std::cerr << e.what() << "\n"; return; }
    }

    for(int g=0; g<games; ++g){
        owen2::Position pos;
        const char* START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        if(book && !book->empty()){
            try { pos.set_fen((*book)[rng() % book->size()]); }
            catch(...) { pos.set_fen(START_FEN); }
        } else {
            pos.set_fen(START_FEN);
        }
        searcher.set_position(pos);
        searcher.new_game();
        int badW=0, badB=0;        // consecutive low-score plies per side (resign tracking)
        int resignedSide = -1;     // -1 none, 0 white resigned, 1 black resigned

        std::vector<SDataRecord> game;
        game.reserve(120);
        std::vector<binpack::ChainEntry> bpgame; // parallel binpack entries
        bpgame.reserve(120);

        // small opening variety: 2 random plies
        for(int r=0; r<2; ++r){
            auto ms = owen2::generate_legal(pos);
            if(ms.empty()) break;
            owen2::Move rm = ms[rng() % ms.size()];
            pos.do_move(rm);
            searcher.set_position(pos);
        }

        for(int ply=0; ply<300; ++ply){
            auto moves = owen2::generate_legal(pos);
            if(moves.empty() || pos.is_draw()) break;

            owen2::search::SearchLimits lim;
            if(depth > 0) lim.depth = depth;
            else lim.movetime_ms = movetime_ms;

            std::atomic<bool> stop{false};
            auto res = searcher.search(lim, stop);
            owen2::Move m = res.bestMove;
            if(!m) m = moves[rng() % moves.size()];

            int side = pos.side_to_move();
            bool resignActive = (resign_cp < 0 && resign_moves > 0);
            if(resignActive){
                if(res.score <= resign_cp){ if(side==owen2::WHITE) ++badW; else ++badB; }
                else                     { if(side==owen2::WHITE) badW=0;   else badB=0; }
            }
            bool goingToResign = resignActive && ((side==owen2::WHITE ? badW : badB) >= resign_moves);

            // NO_PIECE is 12 — distill.py expects 12 for empty, but C++ Piece
            // is an int enum. Cast is correct; the bug was that the struct
            // was zero-initialised with {} which sets empty squares to 0
            // (W_PAWN) inside make_piece histories before set_fen runs.
            // We explicitly skip uninitialized Position copies and validate.
            // Validate the position copy is sane (both kings present). The old
            // check (piece on e1) wrongly discarded every position after white
            // castles or moves his king — losing whole endgames of data.
            bool wk=false, bk=false;
            for(int s=0;s<64;++s){
                owen2::Piece kp = pos.piece_on(s);
                if(kp==owen2::W_KING) wk=true;
                else if(kp==owen2::B_KING) bk=true;
            }
            if(!wk || !bk) continue;
            SDataRecord r{};
            // Ensure empty squares are 12, not 0: piece_on returns 12 for empty,
            // but zero-init of SDataRecord would be 0. Overwrite all 64 explicitly.
            for(int s=0;s<64;++s){
                owen2::Piece pc = pos.piece_on(s);
                r.board[s]=(uint8_t)pc; // 0..11 piece, 12 empty
            }
            r.stm = (uint8_t)pos.side_to_move();
            int sc = std::clamp<int>(res.score, -15000, 15000);
            r.eval = (int16_t)sc;
            r.ply = (uint8_t)std::min(ply, 255);
            r.castling = (uint8_t)pos.castling_rights();
            r.ep = (uint8_t)(pos.ep_square() >= 64 ? 64 : pos.ep_square());
            {
                int fl = owen2::move_flags(m);
                int promo = (fl & owen2::MoveFlag::PROMO) ? (int)owen2::move_promo(m) : 0;
                if (promo < 0 || promo > 4) promo = 0;
                r.move16 = (uint16_t)(owen2::move_from(m) | (owen2::move_to(m) << 6) |
                                     (promo << 12) | ((fl & owen2::MoveFlag::ENPASSANT) ? (1 << 15) : 0));
            }
            game.push_back(r);
            if(bpw){
                binpack::ChainEntry be;
                be.pos = pos;      // snapshot before the move
                be.move = m;       // played move (resign terminal: searched best)
                be.score = sc;
                be.result = 0;
                bpgame.push_back(std::move(be));
            }

            if(goingToResign){
                resignedSide = side;
                break;
            }
            pos.do_move(m);
            searcher.set_position(pos);
        }

        int finalResult=1;
        auto ms = owen2::generate_legal(pos);
        if(ms.empty()){
            finalResult = pos.in_check() ? (pos.side_to_move()==owen2::BLACK ? 2 : 0) : 1;
        } else if(pos.is_draw()){
            finalResult = 1;
        } else if(resignedSide != -1){
            finalResult = (resignedSide==owen2::WHITE ? 0 : 2);
        } else {
            finalResult = 1; // adjudicate long games as draw
        }

        for(size_t i=0;i<game.size();++i){
            int stmSide = game[i].stm; // 0=white to move, 1=black
            uint8_t resFromStm;
            if(finalResult==1) resFromStm=1;
            else {
                bool stmWins = ((finalResult==2) == (stmSide==0));
                resFromStm = stmWins ? 2 : 0;
            }
            game[i].result = resFromStm;
            f.write((char*)&game[i], sizeof(SDataRecord));
        }
        if(bpw && !bpgame.empty()){
            for(size_t i=0;i<bpgame.size();++i){
                int stmSide = bpgame[i].pos.side_to_move();
                int rr = 0; // -1/0/+1 stm perspective
                if(finalResult==2) rr = (stmSide==0) ? 1 : -1;
                else if(finalResult==0) rr = (stmSide==1) ? 1 : -1;
                bpgame[i].result = rr;
            }
            bpw->addGame(bpgame);
        }
        positions += game.size();
        int d = ++done;
        if(d % 500 == 0 || g == games-1){
            std::lock_guard<std::mutex> lk(io_mtx);
            std::cout << "[" << d << "] games, " << positions.load() << " positions\n";
        }
    }
}

int main(int argc, char** argv){
    std::string out="data/sdata.bin";
    std::string net="";
    std::string bookFile="";
    std::string binpackOut=""; // optional ecosystem-format copy of OUR games
    int games=1000;
    int threads=1;
    int movetime_ms=15;
    int depth=-1;
    int resign_cp=-700;
    int resign_moves=3;

    for(int i=1;i<argc;++i){
        std::string a=argv[i];
        auto need=[&](std::string &dst){ if(i+1<argc) dst=argv[++i]; };
        if(a=="--out") need(out);
        else if(a=="--games" && i+1<argc) games=std::stoi(argv[++i]);
        else if(a=="--threads" && i+1<argc) threads=std::stoi(argv[++i]);
        else if(a=="--movetime" && i+1<argc) movetime_ms=std::stoi(argv[++i]);
        else if(a=="--depth" && i+1<argc) depth=std::stoi(argv[++i]);
        else if(a=="--net" && i+1<argc) net=argv[++i];
        else if(a=="--book" && i+1<argc) bookFile=argv[++i];
        else if(a=="--binpack-out" && i+1<argc) binpackOut=argv[++i];
        else if(a=="--resign-cp" && i+1<argc) resign_cp=std::stoi(argv[++i]);
        else if(a=="--resign-moves" && i+1<argc) resign_moves=std::stoi(argv[++i]);
        else if(a=="--help" || a=="-h"){
            std::cout << "Usage: owen2-sdata --games N --out file.bin [--threads T] [--movetime MS|--depth D] [--net file.o2nn] [--book file.epd] [--binpack-out file.binpack] [--resign-cp CP] [--resign-moves M]\n";
            return 0;
        }
    }
    owen2::init_attacks(); owen2::Position::init_zobrist();
    std::filesystem::create_directories(std::filesystem::path(out).parent_path());
    if(!binpackOut.empty())
        std::filesystem::create_directories(std::filesystem::path(binpackOut).parent_path());

    std::vector<std::string> book;
    if(!bookFile.empty()){
        std::ifstream bf(bookFile);
        std::string line;
        while(std::getline(bf,line)){
            if(!line.empty() && line[0]!='#') book.push_back(line);
        }
        if(book.empty()) std::cout << "book " << bookFile << " empty (falling back to startpos)\n";
        else std::cout << "book: " << book.size() << " FENs from " << bookFile << "\n";
    }
    const std::vector<std::string>* bookPtr = book.empty() ? nullptr : &book;

    if(threads < 1) threads=1;
    if(threads > 64) threads=64;
    int per = games / threads;
    int rem = games % threads;

    std::cout << "Owen2 sdata: " << games << " games, " << threads << " threads, "
              << (depth>0 ? ("depth "+std::to_string(depth)) : ("movetime "+std::to_string(movetime_ms)+"ms"))
              << (net.empty()?" (handcrafted)":" net="+net) << "\n"
              << "  resign: " << (resign_cp<0 ? "cp<="+std::to_string(resign_cp)+" for "+std::to_string(resign_moves)+" plies" : "off") << "\n";

    std::atomic<int> done{0};
    std::atomic<uint64_t> positions{0};
    std::mutex io_mtx;
    auto t0 = std::chrono::steady_clock::now();

    std::vector<std::thread> pool;
    for(int t=0; t<threads; ++t){
        int g = per + (t < rem ? 1 : 0);
        if(g==0) continue;
        pool.emplace_back(play_games, t, g, movetime_ms, depth, net, out, binpackOut, bookPtr,
                          resign_cp, resign_moves,
                          std::ref(done), std::ref(positions), std::ref(io_mtx));
    }
    for(auto &th: pool) th.join();

    // merge parts (sdata; binpack parts are each valid chunk streams, concat is valid)
    auto merge_parts = [&](const std::string& dst){
        std::ofstream fout(dst, std::ios::binary);
        std::vector<char> buf(1<<20);
        for(int t=0; t<threads; ++t){
            std::string part = dst + ".part" + std::to_string(t);
            if(!std::filesystem::exists(part)) continue;
            std::ifstream fin(part, std::ios::binary);
            while(fin){
                fin.read(buf.data(), buf.size());
                auto n = fin.gcount();
                if(n>0) fout.write(buf.data(), n);
            }
            std::filesystem::remove(part);
        }
    };
    merge_parts(out);
    if(!binpackOut.empty()) merge_parts(binpackOut);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t0).count();
    uint64_t posN = positions.load();
    double gps = ms ? (done.load()*1000.0/ms) : 0;
    std::cout << "Done: " << done.load() << " games, " << posN << " positions in " << ms/1000.0 << "s"
              << " (" << gps << " games/s, ~" << (posN*1000/ms) << " pos/s)\n";
    std::cout << "Wrote " << out << " (" << std::filesystem::file_size(out) << " bytes)\n";
    if(!binpackOut.empty() && std::filesystem::exists(binpackOut))
        std::cout << "Wrote " << binpackOut << " (" << std::filesystem::file_size(binpackOut) << " bytes)\n";
    // estimate
    double needStockfish = 3e9; // positions
    std::cout << "Progress to ~Beat Stockfish ballpark (~1B pos): " << (posN/1e9*100) << "% this run, "
              << (posN>0? (1e9/posN):0) << "x more runs to 1B\n";
    return 0;
}
