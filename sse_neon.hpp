/*******************************************************************************
* SIMD command
* Licensed under the Simplified BSD License [see external/bsd.txt]
*******************************************************************************/

#ifndef _SIMD_HPP_
#define _SIMD_HPP_


#if __GNUC__
#define FORCE_INLINE                    inline __attribute__((always_inline))
#else
#define FORCE_INLINE                    inline
#endif

#if __GNUC__
#define ALIGN_DATA                      __attribute__((aligned(16)))
#else
#define ALIGN_DATA                      __declspec(align(16))
#endif

#if defined(ANDROID)
#define simdqf float32x4_t
#define simdqi int32x4_t
#else
#define simdqf __m128
#define simdqi __m128i
#endif


#if  defined(ANDROID)
#include <arm_neon.h>

#define RETf inline float32x4_t
#define RETi inline int32x4_t

typedef union {
    float32x4_t f32;
    int32x4_t i32;
    uint32x4_t ui32;
} simdq;

// set, load and store values
RETf SET(const float x) {
    return vdupq_n_f32(x);
}

RETf SET(float x, float y, float z, float w) {
    float ALIGN_DATA data[4] = { x, y, z, w };
    return vld1q_f32(data);
}

RETi SET(const int x) {
    return vdupq_n_s32(x);
}


RETf LD(const float &x) {
    return vld1q_f32(&x);
}
RETf LDu(const float &x) {
    return vld1q_f32(&x);
}

RETf STR(float &x, const simdqf y) {
    vst1q_f32(&x, y);
    return y;
}

RETi STR(int &x, const simdqi y) {
    vst1q_s32(&x, y);
    return y;
}

RETf STR1(float &x, const simdqf y) {
    vst1q_lane_f32(&x, y, 0);
    return y;
}

RETf STRu(float &x, const simdqf y) {
    vst1q_f32(&x, y);
    return y;
}
RETf STR(float &x, const float y) {
    return STR(x,SET(y));
}

// arithmetic operators
RETi ADD(const simdqi x, const simdqi y) {
    return vaddq_s32(x, y);
}

RETf ADD(const simdqf x, const simdqf y) {
    return vaddq_f32(x, y);
}

RETf ADD(const simdqf x, const simdqf y, const simdqf z) {
    return ADD(ADD(x,y),z);
}

RETf ADD(const simdqf a, const simdqf b, const simdqf c, const simdqf &d) {
    return ADD(ADD(ADD(a,b),c),d);
}

RETf SUB(const simdqf x, const simdqf y) {
    return vsubq_f32(x, y);
}

RETf MUL(const simdqf x, const simdqf y) {
    return vmulq_f32(x, y);
}

RETf MUL(const simdqf x, const float y) {
    return MUL(x,SET(y));
}

RETf MUL(const float x, const simdqf y) {
    return MUL(SET(x),y);
}

RETf INC(simdqf &x, const simdqf y) {
    return x = ADD(x, y);
}

RETf INC(float &x, const simdqf y) {
    simdqf t = ADD(LD(x),y);
    return STR(x,t);
}

RETf DEC(simdqf &x, const simdqf y) {
    return x = SUB(x,y);
}

RETf DEC(float &x, const simdqf y) {
    simdqf t = SUB(LD(x),y);
    return STR(x,t);
}

RETf MIN(const simdqf x, const simdqf y) {
    return vminq_f32(x, y);
}

RETf MAX(const simdqf x, const simdqf y) {
    return vmaxq_f32(x, y);
}

RETf RCP(const simdqf x) {
    simdqf recip = vrecpeq_f32(x);
    recip = vmulq_f32(recip, vrecpsq_f32(recip, x));
    return recip;
}

RETf SQRT(const simdqf x) {
    return vsqrtq_f32(x);
}

// conversion operators
RETf CVT(const simdqi x) {
    return vcvtq_f32_s32(x);
}
RETi CVT(const simdqf x) {
    return vcvtq_s32_f32(x);
}



// logical operators
// arm is not perfer to the float and or xor
RETf AND(const simdqf x, const simdqf y) {
    simdq ix, iy, ret;
    ix.f32 = x;
    iy.f32 = y;
    ret.i32 = vandq_s32(ix.i32, iy.i32);
    return ret.f32;
}

RETi AND(const simdqi x, const simdqi y) {
    return vandq_s32(x, y);
}

const simdqi reverse = SET(-1);
RETf ANDNOT(const simdqf x, const simdqf y) {
    simdq ix, iy, ret;
    ix.f32 = x;
    iy.f32 = y;
    ix.i32 = veorq_s32(ix.i32, reverse);
    ret.i32 = vandq_s32(ix.i32, iy.i32);
    return ret.f32;
}

RETf OR(const simdqf x, const simdqf y) {
    simdq ix, iy, ret;
    ix.f32 = x;
    iy.f32 = y;
    ret.i32 = vorrq_s32(ix.i32, iy.i32);
    return ret.f32;
}

RETf XOR(const simdqf x, const simdqf y) {
    simdq ix, iy, ret;
    ix.f32 = x;
    iy.f32 = y;
    ret.i32 = veorq_s32(ix.i32, iy.i32);
    return ret.f32;
}

// comparison operators
RETf CMPGT(const simdqf x, const simdqf y) {
    simdq ret;
    ret.ui32 = vcgtq_f32(x, y);
    return ret.f32;
}
RETf CMPLT(const simdqf x, const simdqf y) {
    simdq ret;
    ret.ui32 = vcltq_f32(x, y);
    return ret.f32;
}

RETi CMPGT(const simdqi x, const simdqi y) {
    simdq ret;
    ret.ui32 = vcgtq_s32(x, y);
    return ret.i32;
}

RETi CMPLT(const simdqi x, const simdqi y) {
    simdq ret;
    ret.ui32 = vcltq_s32(x, y);
    return ret.i32;
}

RETf RCPSQRT(const simdqf x) {
    simdqf e = vrsqrteq_f32(x);
    e = vmulq_f32(e, vrsqrtsq_f32(x, vmulq_f32(e, e)));
    e = vmulq_f32(e, vrsqrtsq_f32(x, vmulq_f32(e, e)));
    return e;
}


#undef RETf
#undef RETi

#else

#include <emmintrin.h> // SSE2:<e*.h>, SSE3:<p*.h>, SSE4:<s*.h>

#define RETf inline __m128
#define RETi inline __m128i

// set, load and store values
RETf SET(const float &x) { return _mm_set1_ps(x); }

RETf SET(float x, float y, float z, float w) { return _mm_set_ps(x, y, z, w); }

RETi SET(const int &x) { return _mm_set1_epi32(x); }

RETf LD(const float &x) { return _mm_load_ps(&x); }

RETf LDu(const float &x) { return _mm_loadu_ps(&x); }

RETf STR(float &x, const simdqf y) {
  _mm_store_ps(&x, y);
  return y;
}

RETf STR1(float &x, const simdqf y) {
  _mm_store_ss(&x, y);
  return y;
}

RETf STRu(float &x, const simdqf y) {
  _mm_storeu_ps(&x, y);
  return y;
}

RETf STR(float &x, const float y) { return STR(x, SET(y)); }

// arithmetic operators
RETi ADD(const simdqi x, const simdqi y) { return _mm_add_epi32(x, y); }

RETf ADD(const simdqf x, const simdqf y) { return _mm_add_ps(x, y); }

RETf ADD(const simdqf x, const simdqf y, const simdqf z) {
  return ADD(ADD(x, y), z);
}

RETf ADD(const simdqf a, const simdqf b, const simdqf c, const simdqf &d) {
  return ADD(ADD(ADD(a, b), c), d);
}

RETf SUB(const simdqf x, const simdqf y) { return _mm_sub_ps(x, y); }

RETf MUL(const simdqf x, const simdqf y) { return _mm_mul_ps(x, y); }

RETf MUL(const simdqf x, const float y) { return MUL(x, SET(y)); }

RETf MUL(const float x, const simdqf y) { return MUL(SET(x), y); }

RETf INC(simdqf &x, const simdqf y) { return x = ADD(x, y); }

RETf INC(float &x, const simdqf y) {
  simdqf t = ADD(LD(x), y);
  return STR(x, t);
}

RETf DEC(simdqf &x, const simdqf y) { return x = SUB(x, y); }

RETf DEC(float &x, const simdqf y) {
  simdqf t = SUB(LD(x), y);
  return STR(x, t);
}

RETf MIN(const simdqf x, const simdqf y) { return _mm_min_ps(x, y); }

RETf RCP(const simdqf x) { return _mm_rcp_ps(x); }

RETf RCPSQRT(const simdqf x) { return _mm_rsqrt_ps(x); }

// logical operators
RETf AND(const simdqf x, const simdqf y) { return _mm_and_ps(x, y); }

RETi AND(const simdqi x, const simdqi y) { return _mm_and_si128(x, y); }

RETf ANDNOT(const simdqf x, const simdqf y) { return _mm_andnot_ps(x, y); }

RETf OR(const simdqf x, const simdqf y) { return _mm_or_ps(x, y); }

RETf XOR(const simdqf x, const simdqf y) { return _mm_xor_ps(x, y); }

RETi XOR(const simdqi x, const simdqi y) { return _mm_xor_si128(x, y); }

// comparison operators
RETf CMPGT(const simdqf x, const simdqf y) { return _mm_cmpgt_ps(x, y); }

RETf CMPLT(const simdqf x, const simdqf y) { return _mm_cmplt_ps(x, y); }

RETi CMPGT(const simdqi x, const simdqi y) { return _mm_cmpgt_epi32(x, y); }

RETi CMPLT(const simdqi x, const simdqi y) { return _mm_cmplt_epi32(x, y); }

// conversion operators
RETf CVT(const simdqi x) { return _mm_cvtepi32_ps(x); }

RETi CVT(const simdqf x) { return _mm_cvttps_epi32(x); }

#undef RETf
#undef RETi
#endif
#endif
