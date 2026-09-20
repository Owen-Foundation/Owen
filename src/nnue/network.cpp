#include "network.h"
#include "sf_bridge.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace owen2::nnue {

Network g_network;

#if defined(__AVX2__)
// Dot product: a = uint8 (0..127 stored as int8), b = int8.
// Uses maddubs (unsigned x signed -> int16 pair sums) then madd -> int32.
inline int32_t dot_u8_s8_avx2(const int8_t* a, const int8_t* b, int n) {
    __m256i acc = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i pair = _mm256_maddubs_epi16(va, vb);      // 16 x int16
        __m256i quad = _mm256_madd_epi16(pair, ones);     // 8 x int32
        acc = _mm256_add_epi32(acc, quad);
    }
    // horizontal sum
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    int32_t total = _mm_cvtsi128_si32(s);
    for (; i < n; ++i) total += (int32_t)a[i] * (int32_t)b[i];
    return total;
}
#endif

#if defined(__ARM_NEON)
// Dot product: a = uint8 (0..127 stored as int8), b = int8.
// Signed 8x8 -> 16 widening multiply, pairwise-accumulated to int32.
// Exact integer math: identical totals to the scalar loop.
inline int32_t dot_u8_s8_neon(const int8_t* a, const int8_t* b, int n) {
    int32x4_t acc = vdupq_n_s32(0);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);
        int16x8_t lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
        int16x8_t hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
        acc = vpadalq_s16(acc, lo);
        acc = vpadalq_s16(acc, hi);
    }
    int32x2_t s2 = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    int32_t total = vget_lane_s32(vpadd_s32(s2, s2), 0);
    for (; i < n; ++i) total += (int32_t)a[i] * (int32_t)b[i];
    return total;
}
#endif

int Network::forward(const std::array<int16_t,H>& acc) const {
    std::array<int8_t, H> h0{};
    for(int i=0;i<H;++i){
        int32_t v = (int32_t)acc[i] + feature_bias[i];
        v >>= 6;
        v = std::clamp<int32_t>(v, 0, 127);
        h0[i] = int8_t(v);
    }
    // 1024 -> 16
    std::array<int16_t, L1> l1{};
    for(int o=0;o<L1;++o){
        int32_t s = l1_bias[o];
#if defined(__AVX2__)
        s += dot_u8_s8_avx2(h0.data(), (const int8_t*)&l1_weights[o*H], H);
#elif defined(__ARM_NEON)
        s += dot_u8_s8_neon(h0.data(), (const int8_t*)&l1_weights[o*H], H);
#else
        for(int i=0;i<H;++i) s += (int32_t)h0[i] * (int32_t)l1_weights[o*H + i];
#endif
        s >>= 6;
        s = std::clamp<int32_t>(s, 0, 127);
        l1[o] = int16_t(s);
    }
    std::array<int16_t, L2> l2{};
    hidden_l2_from_l1(l1, l2);
    int32_t out = out_bias;
    for(int i=0;i<L2;++i) out += (int32_t)l2[i] * (int32_t)out_weights[i];
    out >>= 6;
    return (int)out;
}

void Network::hidden_l2_from_l1(const std::array<int16_t,L1>& l1, std::array<int16_t,L2>& l2) const {
    for(int o=0;o<L2;++o){
        int32_t s = l2_bias[o];
        for(int i=0;i<L1;++i) s += (int32_t)l1[i] * (int32_t)l2_weights[o*L1 + i];
        s >>= 6;
        s = std::clamp<int32_t>(s, 0, 127);
        l2[o] = int16_t(s);
    }
}

void Network::hidden_l2(const std::array<int16_t,H>& acc, std::array<int16_t,L2>& l2) const {
    std::array<int8_t, H> h0{};
    for(int i=0;i<H;++i){
        int32_t v = (int32_t)acc[i] + feature_bias[i];
        v >>= 6;
        v = std::clamp<int32_t>(v, 0, 127);
        h0[i] = int8_t(v);
    }
    std::array<int16_t, L1> l1{};
    for(int o=0;o<L1;++o){
        int32_t s = l1_bias[o];
#if defined(__AVX2__)
        s += dot_u8_s8_avx2(h0.data(), (const int8_t*)&l1_weights[o*H], H);
#elif defined(__ARM_NEON)
        s += dot_u8_s8_neon(h0.data(), (const int8_t*)&l1_weights[o*H], H);
#else
        for(int i=0;i<H;++i) s += (int32_t)h0[i] * (int32_t)l1_weights[o*H + i];
#endif
        s >>= 6;
        s = std::clamp<int32_t>(s, 0, 127);
        l1[o] = int16_t(s);
    }
    hidden_l2_from_l1(l1, l2);
}

int Network::policy_index(Move m) {
    int fr = move_from(m), to = move_to(m);
    int promo = is_promo(m) ? (int)move_promo(m) : 0;
    if(promo >= 1 && promo <= 4) return 4096 + (promo - 1) * 64 + to;
    return fr * 64 + to;
}

float Network::policy_logit(const std::array<int16_t,L2>& l2, int polIdx) const {
    if(!has_policy || polIdx < 0 || polIdx >= NPOL) return 0.0f;
    int32_t s = pol_bias[polIdx];
    const int8_t* w = pol_weights.data() + (size_t)polIdx * L2;
    for(int i=0;i<L2;++i) s += (int32_t)l2[i] * (int32_t)w[i];
    return (float)s / 64.0f;
}

int Network::evaluate(const Position& pos, Accumulator& acc) const {
    if(is_sf_net) return sf_nnue::evaluate(pos);
    if(!loaded) return evaluate_handcrafted(pos);
    refresh_accumulator(pos, acc, feature_weights.data());
    const auto& a = (pos.side_to_move()==WHITE) ? acc.white : acc.black;
    int v = forward(a);
    if(v > 15000) v=15000; if(v < -15000) v=-15000;
    return v;
}
int Network::evaluate_acc(const Position& pos, const Accumulator& acc) const {
    if(is_sf_net) return sf_nnue::evaluate(pos);
    if(!loaded) return evaluate_handcrafted(pos);
    const auto& a = (pos.side_to_move()==WHITE) ? acc.white : acc.black;
    int v = forward(a);
    if(v > 15000) v=15000; if(v < -15000) v=-15000;
    return v;
}
int Network::evaluate(const Position& pos) const {
    if(is_sf_net) return sf_nnue::evaluate(pos);
    if(!loaded) return evaluate_handcrafted(pos);
    Accumulator acc{};
    refresh_accumulator(pos, acc, feature_weights.data());
    const auto& a = (pos.side_to_move()==WHITE) ? acc.white : acc.black;
    int v = forward(a);
    if(v > 15000) v=15000; if(v < -15000) v=-15000;
    return v;
}

int Network::evaluate_handcrafted(const Position& pos) const {
    const int mat[6]={100,320,330,500,900,0};
    // PST from White's view; mirrored for Black via s^56. Values are
    // deliberately sharper than before so startpos ties break to Nf3/Nc3/e4
    // even when the NN is unloaded/untrained — and to keep MTS prior sane.
    const int pst_pawn[64] = {
         0, 0, 0, 0, 0, 0, 0, 0,        // rank 8
        50,50,50,50,50,50,50,50,         // rank 7
        30,30,40,50,50,40,30,30,         // rank 6 — keep structure but don't push f-pawn
         5, 5,12,28,28,12, 5, 5,         // rank 5
         0, 0, 8,22,22, 8, 0, 0,         // rank 4 — e4/d4 rewarded
         5, 5, 8,14,14, 8, 5, 5,         // rank 3
         5, 5, 2,-15,-15,2, 5, 5,        // rank 2 — e2/d2 penalized staying, f2 strongly
         0, 0, 0, 0, 0, 0, 0, 0
    };
    // rank 2 f-file special: f2 pawn on starting square gets -25 total vs e2 -15,
    // so 1.f3 (-28 swing) is strictly worse than 1.e4 (+22 swing) and 1.Nf3.
    const int pst_knight[64] = {
        -55,-40,-30,-30,-30,-30,-40,-55,
        -40,-20,  2,  4,  4,  2,-20,-40,
        -30,  2,14,18,18,14,  2,-30,
        -30,  6,18,26,26,18,  6,-30,
        -30,  2,18,26,26,18,  2,-30,
        -30,  8,14,20,20,14,  8,-30,
        -40,-20,  2,  8,  8,  2,-20,-40,
        -55,-40,-30,-30,-30,-30,-40,-55
    };
    // Central development bonus: knights on f3/c3 etc. already score high;
    // add explicit king-safety penalty if f-pawn has moved before development.
    auto king_moved_penalty = [&]()->int{
        int pen=0;
        // if f-pawn gone from f2 before any knight left g1/b1, penalize
        bool kf = (pos.piece_on(make_square(5,1))==NO_PIECE); // f2 empty
        bool knGone = (pos.piece_on(make_square(6,0))!=W_KNIGHT || pos.piece_on(make_square(1,0))!=W_KNIGHT);
        // penalize from White's perspective; negated later for Black stm
        if(kf && !knGone) pen -= 55;
        // mirror for black
        bool kfB = (pos.piece_on(make_square(5,6))==NO_PIECE);
        bool knGoneB = (pos.piece_on(make_square(6,7))!=B_KNIGHT || pos.piece_on(make_square(1,7))!=B_KNIGHT);
        if(kfB && !knGoneB) pen += 55;
        return pen;
    };
    int score=0;
    for(int s=0;s<64;++s){
        Piece p=pos.piece_on(Square(s));
        if(p==NO_PIECE) continue;
        int v = mat[type_of(p)];
        Color c = color_of(p);
        if(type_of(p)==PAWN){
            int sq2 = c==WHITE ? s : (s ^ 56);
            v += pst_pawn[sq2];
            // extra f-file anti-weakness on top of PST so leaf eval hates f3/f6
            if(file_of(Square(s))==5 && ((c==WHITE && rank_of(Square(s))==2) || (c==BLACK && rank_of(Square(s))==5))){
                // pawn on f3/f6: additional -35
                v -= 35;
            }
        } else if(type_of(p)==KNIGHT){
            int sq2 = c==WHITE ? s : (s ^ 56);
            v += pst_knight[sq2];
        }
        if(c==WHITE) score += v; else score -= v;
    }
    score += king_moved_penalty();
    if(pos.side_to_move()==BLACK) score = -score;
    return score;
}

bool Network::load(const std::string& path){
    if(sf_nnue::load_sf_net(path)) {
        loaded = true;
        is_sf_net = true;
        has_policy = false;
        return true;
    }
    std::ifstream f(path, std::ios::binary);
    if(!f) return false;
    std::vector<unsigned char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    bool ok = load_from_memory(buf.data(), buf.size());
    if(ok) is_sf_net = false;
    return ok;
}
bool Network::load_from_memory(const unsigned char* data, size_t size){
    if(size < 12) return false;
    if(std::memcmp(data,"O2NN",4)!=0) return false;
    uint32_t ver, h;
    std::memcpy(&ver, data+4, 4); std::memcpy(&h, data+8, 4);
    if(h != (uint32_t)H) return false;
    if(ver!=1 && ver!=2 && ver!=3) return false;
    size_t off=12;
    auto need = [&](size_t n){ return off + n <= size; };
    feature_weights.resize((size_t)INPUT_SIZE * H);
    size_t fw = feature_weights.size()*sizeof(int16_t);
    if(!need(fw)) return false; std::memcpy(feature_weights.data(), data+off, fw); off+=fw;
    if(!need(H*sizeof(int16_t))) return false; std::memcpy(feature_bias.data(), data+off, H*sizeof(int16_t)); off+=H*sizeof(int16_t);
    if(!need(H*L1)) return false; std::memcpy(l1_weights.data(), data+off, H*L1); off+=H*L1;
    if(!need(L1*sizeof(int16_t))) return false; std::memcpy(l1_bias.data(), data+off, L1*sizeof(int16_t)); off+=L1*sizeof(int16_t);
    if(!need(L1*L2)) return false; std::memcpy(l2_weights.data(), data+off, L1*L2); off+=L1*L2;
    if(!need(L2*sizeof(int16_t))) return false; std::memcpy(l2_bias.data(), data+off, L2*sizeof(int16_t)); off+=L2*sizeof(int16_t);
    if(!need(L2)) return false; std::memcpy(out_weights.data(), data+off, L2); off+=L2;
    if(!need(sizeof(int16_t))) return false; std::memcpy(&out_bias, data+off, sizeof(int16_t)); off+=sizeof(int16_t);
    has_policy = false;
    pol_weights.clear(); pol_bias.clear();
    if(ver == 3){
        size_t pw = (size_t)Network::NPOL * L2;
        size_t pb = (size_t)Network::NPOL * sizeof(int16_t);
        if(!need(pw + pb)) return false;
        pol_weights.resize(pw);
        std::memcpy(pol_weights.data(), data+off, pw); off+=pw;
        pol_bias.resize(Network::NPOL);
        std::memcpy(pol_bias.data(), data+off, pb); off+=pb;
        has_policy = true;
    }
    loaded=true; return true;
}
bool Network::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if(!f) return false;
    f.write("O2NN",4);
    uint32_t ver=2,h=H; f.write((char*)&ver,4); f.write((char*)&h,4);
    // ensure weights allocated
    if(feature_weights.size() != (size_t)INPUT_SIZE*H) return false;
    f.write((char*)feature_weights.data(), feature_weights.size()*sizeof(int16_t));
    f.write((char*)feature_bias.data(), H*sizeof(int16_t));
    f.write((char*)l1_weights.data(), H*L1);
    f.write((char*)l1_bias.data(), L1*sizeof(int16_t));
    f.write((char*)l2_weights.data(), L1*L2);
    f.write((char*)l2_bias.data(), L2*sizeof(int16_t));
    f.write((char*)out_weights.data(), L2);
    f.write((char*)&out_bias, sizeof(int16_t));
    return !!f;
}

} // namespace owen2::nnue
