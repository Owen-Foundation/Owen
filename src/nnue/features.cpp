#include "features.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace owen2::nnue {

inline void add_weights_avx(int16_t* dst, const int16_t* src) {
#if defined(__AVX2__)
    // 1024 int16 = 64 x __m256i (16 lanes). Wraps mod 2^16 like scalar.
    for (int i = 0; i < HIDDEN_SIZE; i += 16) {
        __m256i d = _mm256_loadu_si256((const __m256i*)(dst + i));
        __m256i s = _mm256_loadu_si256((const __m256i*)(src + i));
        __m256i r = _mm256_add_epi16(d, s);
        _mm256_storeu_si256((__m256i*)(dst + i), r);
    }
#else
    for (int i = 0; i < HIDDEN_SIZE; ++i) dst[i] += src[i];
#endif
}

int feature_index(Color perspective, Square kingSq, Piece piece, Square pieceSq){
    auto orient = [&](Square s)->Square { return perspective==BLACK ? Square(s ^ 56) : s; };
    Square k = orient(kingSq);
    Square ps = orient(pieceSq);
    int pc = (int)piece;
    if(pc==W_KING || pc==B_KING) return -1;
    int pc10 = pc < 6 ? pc : (pc-6)+5;
    if(pc10<0||pc10>=10) return -1;
    return pc10*4096 + k*64 + ps; // 0..40959
}
int threat_index(Color perspective, Square kingSq, Piece piece, Square pieceSq){
    int base = feature_index(perspective, kingSq, piece, pieceSq);
    if(base < 0) return -1;
    return HALFKP_SIZE + base; // 40960..81919
}

void refresh_accumulator(const Position& pos, Accumulator& acc, const int16_t* weights){
    acc.white.fill(0); acc.black.fill(0);
    Square wk = pos.king_sq(WHITE), bk = pos.king_sq(BLACK);
    for(int s=0;s<64;++s){
        Piece p = pos.piece_on(Square(s));
        if(p==NO_PIECE) continue;
        if(type_of(p)==KING) continue;
        int idxW = feature_index(WHITE, wk, p, Square(s));
        int idxB = feature_index(BLACK, bk, p, Square(s));
        if(idxW>=0){
            const int16_t* w = weights + idxW * HIDDEN_SIZE;
            add_weights_avx(acc.white.data(), w);
        }
        if(idxB>=0){
            const int16_t* w = weights + idxB * HIDDEN_SIZE;
            add_weights_avx(acc.black.data(), w);
        }
        // Threat inputs: + second plane if piece is threatened
        if(is_threatened(pos, Square(s), WHITE) && idxW>=0){
            int tW = threat_index(WHITE, wk, p, Square(s));
            const int16_t* w = weights + tW * HIDDEN_SIZE;
            add_weights_avx(acc.white.data(), w);
        }
        if(is_threatened(pos, Square(s), BLACK) && idxB>=0){
            int tB = threat_index(BLACK, bk, p, Square(s));
            const int16_t* w = weights + tB * HIDDEN_SIZE;
            add_weights_avx(acc.black.data(), w);
        }
    }
    acc.computed_white=acc.computed_black=true;
}

} // namespace owen2::nnue
