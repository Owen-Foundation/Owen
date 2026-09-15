#include "position.h"
#include <sstream>
#include <random>
#include <cctype>
#include <cassert>

namespace owen2 {

uint64_t Position::ZobristPiece[12][64]{};
uint64_t Position::ZobristSide=0;
uint64_t Position::ZobristCastle[16]{};
uint64_t Position::ZobristEP[8]{};
bool Position::zobristInit_=false;

void Position::init_zobrist(){
    if(zobristInit_) return;
    std::mt19937_64 rng(0x4F3A9E2D1C7B5A68ULL);
    for(int p=0;p<12;++p) for(int s=0;s<64;++s) ZobristPiece[p][s]=rng();
    ZobristSide=rng();
    for(int i=0;i<16;++i) ZobristCastle[i]=rng();
    for(int i=0;i<8;++i) ZobristEP[i]=rng();
    zobristInit_=true;
}

void Position::put_piece(Piece p, Square s){
    board_[s]=p;
    Bitboard bb=sq_bb(s);
    byColor_[color_of(p)] |= bb;
    byColorType_[color_of(p)][type_of(p)] |= bb;
    byType_[type_of(p)] |= bb;
    if(type_of(p)==KING) kingSq_[color_of(p)]=s;
    st_.key ^= ZobristPiece[p][s];
}
void Position::remove_piece(Square s){
    Piece p=board_[s];
    assert(p!=NO_PIECE);
    Bitboard bb=sq_bb(s);
    byColor_[color_of(p)] &= ~bb;
    byColorType_[color_of(p)][type_of(p)] &= ~bb;
    byType_[type_of(p)] &= ~bb;
    st_.key ^= ZobristPiece[p][s];
    board_[s]=NO_PIECE;
}

void Position::update_checkers(){
    Square ksq = kingSq_[stm_];
    Color them = ~stm_;
    Bitboard occ = occupancy();
    Bitboard c = 0;
    c |= KnightAttacks[ksq] & pieces(them, KNIGHT);
    c |= KingAttacks[ksq] & pieces(them, KING);
    Bitboard pawnAtt = (stm_==WHITE) ? PawnAttacksBlack[ksq] : PawnAttacksWhite[ksq];
    c |= pawnAtt & pieces(them, PAWN);
    c |= bishop_attacks(ksq, occ) & (pieces(them, BISHOP) | pieces(them, QUEEN));
    c |= rook_attacks(ksq, occ)   & (pieces(them, ROOK)   | pieces(them, QUEEN));
    st_.checkers = c;
}

Bitboard Position::attackers_to(Square s, Bitboard occ) const {
    Bitboard att=0;
    att |= PawnAttacksWhite[s] & pieces(BLACK, PAWN);
    att |= PawnAttacksBlack[s] & pieces(WHITE, PAWN);
    att |= KnightAttacks[s] & (pieces(WHITE,KNIGHT)|pieces(BLACK,KNIGHT));
    att |= KingAttacks[s]   & (pieces(WHITE,KING)|pieces(BLACK,KING));
    att |= bishop_attacks(s,occ) & (byType_[BISHOP]|byType_[QUEEN]);
    att |= rook_attacks(s,occ)   & (byType_[ROOK]|byType_[QUEEN]);
    return att;
}
bool Position::square_attacked(Square s, Color by) const {
    Bitboard occ=occupancy();
    if(by==WHITE){
        if(PawnAttacksBlack[s] & pieces(WHITE,PAWN)) return true;
    } else {
        if(PawnAttacksWhite[s] & pieces(BLACK,PAWN)) return true;
    }
    if(KnightAttacks[s] & pieces(by,KNIGHT)) return true;
    if(KingAttacks[s] & pieces(by,KING)) return true;
    if(bishop_attacks(s,occ) & (pieces(by,BISHOP)|pieces(by,QUEEN))) return true;
    if(rook_attacks(s,occ)   & (pieces(by,ROOK)|pieces(by,QUEEN))) return true;
    return false;
}

void Position::set_fen(const std::string& fen){
    init_zobrist();
    for(int s=0;s<64;++s) board_[s]=NO_PIECE;
    byColor_[0]=byColor_[1]=0;
    for(int c=0;c<2;++c) for(int pt=0;pt<6;++pt) byColorType_[c][pt]=0;
    for(int pt=0;pt<6;++pt) byType_[pt]=0;
    st_={}; ply_=0; history_.clear();

    std::istringstream iss(fen);
    std::string bstr, stm, castle, ep, half, full;
    iss >> bstr >> stm >> castle >> ep >> half >> full;

    int rank=7, file=0;
    bool badFEN=false;
    for(char ch: bstr){
        if(ch=='/'){ rank--; file=0; if(rank<0||file>8) badFEN=true; continue; }
        if(std::isdigit(ch)){
            int d=ch-'0';
            if(d<1||d>8) badFEN=true;
            file += d;
            if(file>8) badFEN=true;
            continue;
        }
        Piece p=NO_PIECE;
        switch(ch){
            case 'P': p=W_PAWN; break; case 'N': p=W_KNIGHT; break;
            case 'B': p=W_BISHOP; break; case 'R': p=W_ROOK; break;
            case 'Q': p=W_QUEEN; break; case 'K': p=W_KING; break;
            case 'p': p=B_PAWN; break; case 'n': p=B_KNIGHT; break;
            case 'b': p=B_BISHOP; break; case 'r': p=B_ROOK; break;
            case 'q': p=B_QUEEN; break; case 'k': p=B_KING; break;
            default: badFEN=true; continue;
        }
        if(p==NO_PIECE || file>=8 || rank<0 || rank>=8) { badFEN=true; continue; }
        Square sq=make_square(file,rank);
        board_[sq]=p;
        Bitboard bb=sq_bb(sq);
        byColor_[color_of(p)] |= bb;
        byColorType_[color_of(p)][type_of(p)] |= bb;
        byType_[type_of(p)] |= bb;
        if(type_of(p)==KING) kingSq_[color_of(p)]=sq;
        file++;
    }
    if(badFEN || rank!=0){
        // Fall back to startpos rather than leaving a corrupt board
        for(int s=0;s<64;++s) board_[s]=NO_PIECE;
        byColor_[0]=byColor_[1]=0;
        for(int c=0;c<2;++c) for(int pt=0;pt<6;++pt) byColorType_[c][pt]=0;
        for(int pt=0;pt<6;++pt) byType_[pt]=0;
        set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        return;
    }
    stm_ = (stm=="w"||stm=="W")?WHITE:BLACK;
    st_.castling=0;
    if(castle!="-") for(char c: castle){
        if(c=='K') st_.castling|=1;
        if(c=='Q') st_.castling|=2;
        if(c=='k') st_.castling|=4;
        if(c=='q') st_.castling|=8;
    }
    if(ep!="-" && ep.size()>=2 && ep[0]>='a'&&ep[0]<='h' && ep[1]>='1'&&ep[1]<='8'){
        int f=ep[0]-'a', r=ep[1]-'1';
        st_.ep_square = make_square(f,r);
    } else st_.ep_square=64;
    // half/full may be absent or non-numeric — don't let stoi throw through to caller
    try { st_.rule50 = half.empty()?0:std::stoi(half); } catch(...){ st_.rule50=0; }
    try { if(!full.empty()) ply_ = std::max(0, (std::stoi(full)-1)*2 + (stm_==BLACK?1:0)); else ply_=0; } catch(...){ ply_=0; }
    (void)full;

    st_.key=0;
    for(int s=0;s<64;++s) if(board_[s]!=NO_PIECE) st_.key ^= ZobristPiece[board_[s]][s];
    if(stm_==BLACK) st_.key ^= ZobristSide;
    st_.key ^= ZobristCastle[st_.castling & 0xF];
    if(st_.ep_square!=64) st_.key ^= ZobristEP[file_of(st_.ep_square)];
    for(int s=0;s<64;++s) st_.board[s]=board_[s];
    update_checkers();
}

std::string Position::fen() const {
    std::string s;
    for(int r=7;r>=0;--r){
        int empty=0;
        for(int f=0;f<8;++f){
            Piece p=board_[make_square(f,r)];
            if(p==NO_PIECE) empty++;
            else {
                if(empty){ s+=char('0'+empty); empty=0; }
                const char ch[]="PNBRQKpnbrqk";
                s+=ch[p];
            }
        }
        if(empty) s+=char('0'+empty);
        if(r) s+='/';
    }
    s += (stm_==WHITE?" w ":" b ");
    std::string cs;
    if(st_.castling&1) cs+='K';
    if(st_.castling&2) cs+='Q';
    if(st_.castling&4) cs+='k';
    if(st_.castling&8) cs+='q';
    if(cs.empty()) cs="-";
    s+=cs; s+=' ';
    if(st_.ep_square==64) s+="-"; else s+=sq_to_str(st_.ep_square);
    s+=" " + std::to_string(st_.rule50) + " " + std::to_string(1 + ply_/2);
    return s;
}

void Position::do_move(Move m){
    // snapshot for undo
    st_.board = [&]{ std::array<Piece,64> a{}; for(int i=0;i<64;++i) a[i]=board_[i]; return a; }();
    // we need to store current st before mutation; copy board already done via st_.board
    // but history needs previous st (with its board). So save st before changing, and keep board snapshot in saved copy.
    // st_.board currently holds current board; history will hold it.
    StateInfo prev = st_;
    // keep board snapshot in prev (already)
    history_.push_back(prev);

    Square from=move_from(m), to=move_to(m);
    int flags=move_flags(m);
    Piece moving = board_[from];
    Piece captured = board_[to];
    Color us = stm_, them = ~us;

    st_.moved = moving;
    st_.captured = captured;
    st_.last_move = m;

    // key: remove castle/ep/side
    st_.key ^= ZobristCastle[st_.castling & 0xF];
    if(st_.ep_square!=64) st_.key ^= ZobristEP[file_of(st_.ep_square)];

    if(captured!=NO_PIECE) remove_piece(to);

    if(flags & MoveFlag::ENPASSANT){
        Square capSq = make_square(file_of(to), rank_of(from));
        remove_piece(capSq);
        st_.captured = make_piece(them, PAWN);
    }

    remove_piece(from);
    Piece placed = moving;
    if(flags & MoveFlag::PROMO) placed = make_piece(us, move_promo(m));
    put_piece(placed, to);

    if(flags & MoveFlag::CASTLING){
        if(file_of(to)==6){
            Square rf=make_square(7, rank_of(from)), rt=make_square(5, rank_of(from));
            Piece rook=board_[rf]; remove_piece(rf); put_piece(rook, rt);
        } else {
            Square rf=make_square(0, rank_of(from)), rt=make_square(3, rank_of(from));
            Piece rook=board_[rf]; remove_piece(rf); put_piece(rook, rt);
        }
    }

    auto clear = [&](Square sq){
        if(sq==make_square(4,0)) st_.castling &= ~(1|2);
        if(sq==make_square(7,0)) st_.castling &= ~1;
        if(sq==make_square(0,0)) st_.castling &= ~2;
        if(sq==make_square(4,7)) st_.castling &= ~(4|8);
        if(sq==make_square(7,7)) st_.castling &= ~4;
        if(sq==make_square(0,7)) st_.castling &= ~8;
    };
    clear(from); clear(to);

    st_.ep_square=64;
    if(flags & MoveFlag::DOUBLE){
        st_.ep_square = make_square(file_of(from), (rank_of(from)+rank_of(to))/2);
        st_.key ^= ZobristEP[file_of(st_.ep_square)];
    }

    if(type_of(moving)==PAWN || captured!=NO_PIECE || (flags&MoveFlag::ENPASSANT)) st_.rule50=0;
    else st_.rule50++;

    stm_ = them;
    st_.key ^= ZobristSide;
    st_.key ^= ZobristCastle[st_.castling & 0xF];

    ply_++;
    // snapshot new board into st_.board
    for(int i=0;i<64;++i) st_.board[i]=board_[i];
    update_checkers();
}

void Position::undo_move(Move m){
    (void)m;
    assert(!history_.empty());
    StateInfo prev = history_.back(); history_.pop_back();
    // restore board from prev snapshot
    // clear current occupancy then rebuild from prev.board
    for(int s=0;s<64;++s) board_[s]=prev.board[s];
    // rebuild bitboards from board
    byColor_[0]=byColor_[1]=0;
    for(int c=0;c<2;++c) for(int pt=0;pt<6;++pt) byColorType_[c][pt]=0;
    for(int pt=0;pt<6;++pt) byType_[pt]=0;
    for(int s=0;s<64;++s) if(board_[s]!=NO_PIECE){
        Piece p=board_[s];
        Bitboard bb=sq_bb(s);
        byColor_[color_of(p)] |= bb;
        byColorType_[color_of(p)][type_of(p)] |= bb;
        byType_[type_of(p)] |= bb;
        if(type_of(p)==KING) kingSq_[color_of(p)]=s;
    }
    stm_ = (stm_==WHITE?BLACK:WHITE);
    st_ = prev;
    ply_--;
}

bool Position::is_draw() const {
    if(st_.rule50 >= 100) return true;
    int cnt=0;
    for(auto &h: history_) if(h.key==st_.key) { cnt++; if(cnt>=2) return true; }
    // insufficient material
    int pcs = popcount(occupancy());
    if(pcs==2) return true;
    if(pcs==3 && (byType_[KNIGHT]|byType_[BISHOP])) return true;
    return false;
}

} // namespace owen2
