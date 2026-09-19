#include "bitboard.h"
#include <cstdio>

namespace owen2 {

std::array<Bitboard,64> KnightAttacks{};
std::array<Bitboard,64> KingAttacks{};
std::array<Bitboard,64> PawnAttacksWhite{};
std::array<Bitboard,64> PawnAttacksBlack{};

static Bitboard sliding_attacks(Square sq, Bitboard occ, const int dirs[4][2], int ndirs){
    Bitboard attacks=0;
    int r0=rank_of(sq), f0=file_of(sq);
    for(int d=0;d<ndirs;++d){
        int r=r0+dirs[d][0], f=f0+dirs[d][1];
        while(r>=0&&r<8&&f>=0&&f<8){
            Square s=make_square(f,r);
            attacks |= sq_bb(s);
            if(occ & sq_bb(s)) break;
            r+=dirs[d][0]; f+=dirs[d][1];
        }
    }
    return attacks;
}

static Bitboard bishop_attacks_naive(Square sq, Bitboard occ){
    const int dirs[4][2]={{1,1},{1,-1},{-1,1},{-1,-1}};
    return sliding_attacks(sq,occ,dirs,4);
}
static Bitboard rook_attacks_naive(Square sq, Bitboard occ){
    const int dirs[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
    return sliding_attacks(sq,occ,dirs,4);
}

// ── Magic bitboards (clean-room) ───────────────────────────────────
// Magics are generated at startup by a fixed-seed PRNG (deterministic,
// zero copied constants), then EVERY table entry is verified against the
// naive slider above. Any mismatch aborts init — never silently wrong.
namespace {
uint64_t mg_rng = 0x9E3779B97F4A7C15ULL;
uint64_t mg_next(){ // splitmix64
    uint64_t z = (mg_rng += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
inline uint64_t mg_sparse(){ return mg_next() & mg_next() & mg_next(); }

Bitboard slider_mask(Square sq, bool bishop){
    Bitboard m = 0;
    int r0=rank_of(sq), f0=file_of(sq);
    const int bd[4][2]={{1,1},{1,-1},{-1,1},{-1,-1}};
    const int rd[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
    const int (*dirs)[2] = bishop ? bd : rd;
    for(int d=0;d<4;++d){
        int r=r0+dirs[d][0], f=f0+dirs[d][1];
        while(r>=0&&r<8&&f>=0&&f<8){
            // The terminal edge square's occupancy never changes the attack
            // set (nothing lies beyond it), so only it is excluded. Squares
            // that merely SIT on an edge (e.g. a2 for a rook on a1) still
            // block rays and must be included.
            int nr=r+dirs[d][0], nf=f+dirs[d][1];
            if(nr<0||nr>=8||nf<0||nf>=8) break;
            m |= sq_bb(make_square(f,r));
            r=nr; f=nf;
        }
    }
    return m;
}

Bitboard set_occ(int index, Bitboard mask){
    Bitboard occ = 0;
    int b = 0;
    while(mask){
        int j = std::countr_zero(mask);
        mask &= mask - 1;
        if(index & (1 << b)) occ |= (1ULL << j);
        ++b;
    }
    return occ;
}

Bitboard find_magic(Square sq, int bits, bool bishop, Bitboard mask){
    Bitboard occs[4096], refs[4096];
    int n = 1 << bits;
    for(int i=0;i<n;++i){
        occs[i] = set_occ(i, mask);
        refs[i] = bishop ? bishop_attacks_naive(sq, occs[i])
                         : rook_attacks_naive(sq, occs[i]);
    }
    static Bitboard used[4096];
    static int stamp[4096] = {};
    static int epoch = 0;
    for(;;){
        Bitboard magic = mg_sparse();
        // Top-6-bits heuristic: good magics spread entropy upward.
        if(popcount((mask * magic) & 0xFF00000000000000ULL) < 6) continue;
        ++epoch;
        bool fail = false;
        for(int i=0;i<n && !fail;++i){
            size_t idx = (size_t)((occs[i] * magic) >> (64 - bits));
            if(stamp[idx] == epoch && used[idx] != refs[i]) fail = true;
            else { stamp[idx] = epoch; used[idx] = refs[i]; }
        }
        if(!fail) return magic;
    }
}

Bitboard b_magic[64], r_magic[64];
Bitboard b_mask[64], r_mask[64];
int b_bits[64], r_bits[64];
Bitboard b_table[64][512];
Bitboard r_table[64][4096];
bool mg_ready = false;

void init_magics(){
    if(mg_ready) return;
    mg_rng = 0x9E3779B97F4A7C15ULL; // deterministic
    for(int sq=0;sq<64;++sq){
        b_mask[sq] = slider_mask(sq, true);
        r_mask[sq] = slider_mask(sq, false);
        b_bits[sq] = popcount(b_mask[sq]);
        r_bits[sq] = popcount(r_mask[sq]);
        // Defensive: 1..9 bishop bits (512 table), 1..12 rook bits (4096).
        // Anything else would shift by >=64 (UB) or overflow the table.
        if(b_bits[sq] < 1 || b_bits[sq] > 9 || r_bits[sq] < 1 || r_bits[sq] > 12){
            std::fprintf(stderr, "FATAL: bad mask bits sq=%d b=%d r=%d\n",
                         sq, b_bits[sq], r_bits[sq]);
            std::abort();
        }
        b_magic[sq] = find_magic(sq, b_bits[sq], true, b_mask[sq]);
        r_magic[sq] = find_magic(sq, r_bits[sq], false, r_mask[sq]);
        int bn = 1 << b_bits[sq], rn = 1 << r_bits[sq];
        for(int i=0;i<bn;++i){
            Bitboard occ = set_occ(i, b_mask[sq]);
            size_t idx = (size_t)((occ * b_magic[sq]) >> (64 - b_bits[sq]));
            b_table[sq][idx] = bishop_attacks_naive(sq, occ);
        }
        for(int i=0;i<rn;++i){
            Bitboard occ = set_occ(i, r_mask[sq]);
            size_t idx = (size_t)((occ * r_magic[sq]) >> (64 - r_bits[sq]));
            r_table[sq][idx] = rook_attacks_naive(sq, occ);
        }
    }
    // Exhaustive self-verification: every square x every occupancy.
    for(int sq=0;sq<64;++sq){
        int bn = 1 << b_bits[sq], rn = 1 << r_bits[sq];
        for(int i=0;i<bn;++i){
            Bitboard occ = set_occ(i, b_mask[sq]);
            size_t idx = (size_t)((occ * b_magic[sq]) >> (64 - b_bits[sq]));
            if(b_table[sq][idx] != bishop_attacks_naive(sq, occ)){
                std::fprintf(stderr, "FATAL: bishop magic mismatch sq=%d\n", sq);
                std::abort();
            }
        }
        for(int i=0;i<rn;++i){
            Bitboard occ = set_occ(i, r_mask[sq]);
            size_t idx = (size_t)((occ * r_magic[sq]) >> (64 - r_bits[sq]));
            if(r_table[sq][idx] != rook_attacks_naive(sq, occ)){
                std::fprintf(stderr, "FATAL: rook magic mismatch sq=%d\n", sq);
                std::abort();
            }
        }
    }
    mg_ready = true;
}
} // namespace

Bitboard bishop_attacks(Square sq, Bitboard occ){
    size_t idx = (size_t)(((occ & b_mask[sq]) * b_magic[sq]) >> (64 - b_bits[sq]));
    return b_table[sq][idx];
}
Bitboard rook_attacks(Square sq, Bitboard occ){
    size_t idx = (size_t)(((occ & r_mask[sq]) * r_magic[sq]) >> (64 - r_bits[sq]));
    return r_table[sq][idx];
}

void init_attacks(){
    for(int sq=0;sq<64;++sq){
        int r=rank_of(sq), f=file_of(sq);
        Bitboard k=0,n=0,pw=0,pb=0;
        // king
        for(int dr=-1;dr<=1;++dr) for(int df=-1;df<=1;++df){
            if(dr==0&&df==0) continue;
            int nr=r+dr,nf=f+df;
            if(nr>=0&&nr<8&&nf>=0&&nf<8) k |= sq_bb(make_square(nf,nr));
        }
        // knight
        const int kd[8][2]={{2,1},{2,-1},{-2,1},{-2,-1},{1,2},{1,-2},{-1,2},{-1,-2}};
        for(auto &d: kd){
            int nr=r+d[0], nf=f+d[1];
            if(nr>=0&&nr<8&&nf>=0&&nf<8) n |= sq_bb(make_square(nf,nr));
        }
        // pawns (attacks FROM square: white pawns attack north)
        if(r<7){
            if(f>0) pw |= sq_bb(make_square(f-1,r+1));
            if(f<7) pw |= sq_bb(make_square(f+1,r+1));
        }
        if(r>0){
            if(f>0) pb |= sq_bb(make_square(f-1,r-1));
            if(f<7) pb |= sq_bb(make_square(f+1,r-1));
        }
        KingAttacks[sq]=k;
        KnightAttacks[sq]=n;
        PawnAttacksWhite[sq]=pw;
        PawnAttacksBlack[sq]=pb;
    }
    init_magics(); // after leaper tables; aborts on any mismatch
}

} // namespace owen2
