/**
 * @file sse_math.h
 *
 * @brief Transcendental math functions for SSE packs.
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_MATH_H_
#define LSIMD_SSE_MATH_H_

#include <light_simd/sse/sse_pack.h>

#if defined(LSIMD_USE_INTEL_SVML) || defined(LSIMD_USE_AMD_LIBM)
#define LSIMD_USE_MATH_FUNCTIONS
#endif

#if defined(LSIMD_USE_INTEL_SVML) && defined(LSIMD_USE_AMD_LIBM)
#error SVML and LIBM cannot be used simultaneously.
#endif


#ifdef LSIMD_USE_INTEL_SVML

// External function prototypes

#define SVML_SSE_F( name ) __svml_##name##f4
#define SVML_SSE_D( name ) __svml_##name##2

#define DECLARE_SVML_SSE_EXTERN1( name ) \
	__m128  SVML_SSE_F(name)( __m128 ); \
	__m128d SVML_SSE_D(name)( __m128d );

#define DECLARE_SVML_SSE_EXTERN2( name ) \
	__m128  SVML_SSE_F(name)( __m128,  __m128  ); \
	__m128d SVML_SSE_D(name)( __m128d, __m128d );

#define LSIMD_SSE_F( name ) SVML_SSE_F( name )
#define LSIMD_SSE_D( name ) SVML_SSE_D( name )

#define LSIMD_HAS_SSE_ANTI_TRIGONO
#define LSIMD_HAS_SSE_HYPERBOLIC
#define LSIMD_HAS_SSE_ANTI_HYPERBOLIC
#define LSIMD_HAS_SSE_HYPOT
#define LSIMD_HAS_SSE_ERF

extern "C"
{
	DECLARE_SVML_SSE_EXTERN1( cbrt )
	DECLARE_SVML_SSE_EXTERN2( pow )
	DECLARE_SVML_SSE_EXTERN2( hypot )

	DECLARE_SVML_SSE_EXTERN1( exp )
	DECLARE_SVML_SSE_EXTERN1( exp2 )
	DECLARE_SVML_SSE_EXTERN1( exp10 )
	DECLARE_SVML_SSE_EXTERN1( expm1 )

	DECLARE_SVML_SSE_EXTERN1( log )
	DECLARE_SVML_SSE_EXTERN1( log2 )
	DECLARE_SVML_SSE_EXTERN1( log10 )
	DECLARE_SVML_SSE_EXTERN1( log1p )

	DECLARE_SVML_SSE_EXTERN1( sin )
	DECLARE_SVML_SSE_EXTERN1( cos )
	DECLARE_SVML_SSE_EXTERN1( tan )

	DECLARE_SVML_SSE_EXTERN1( asin )
	DECLARE_SVML_SSE_EXTERN1( acos )
	DECLARE_SVML_SSE_EXTERN1( atan )
	DECLARE_SVML_SSE_EXTERN2( atan2 )

	DECLARE_SVML_SSE_EXTERN1( sinh )
	DECLARE_SVML_SSE_EXTERN1( cosh )
	DECLARE_SVML_SSE_EXTERN1( tanh )

	DECLARE_SVML_SSE_EXTERN1( asinh )
	DECLARE_SVML_SSE_EXTERN1( acosh )
	DECLARE_SVML_SSE_EXTERN1( atanh )

	DECLARE_SVML_SSE_EXTERN1( erf )
	DECLARE_SVML_SSE_EXTERN1( erfc )
}

#endif  /* LSIMD_USE_SMVL */


#ifdef LSIMD_USE_AMD_LIBM

#define LIBM_SSE_F( name ) amd_vrs4_##name##f
#define LIBM_SSE_D( name ) amd_vrd2_##name

#define DECLARE_LIBM_SSE_EXTERN1( name ) \
	__m128  LIBM_SSE_F(name)( __m128 ); \
	__m128d LIBM_SSE_D(name)( __m128d );

#define DECLARE_LIBM_SSE_EXTERN2( name ) \
	__m128  LIBM_SSE_F(name)( __m128,  __m128  ); \
	__m128d LIBM_SSE_D(name)( __m128d, __m128d );

#define LSIMD_SSE_F( name ) LIBM_SSE_F( name )
#define LSIMD_SSE_D( name ) LIBM_SSE_D( name )

extern "C"
{
	DECLARE_LIBM_SSE_EXTERN1( cbrt )
	DECLARE_LIBM_SSE_EXTERN2( pow )

	DECLARE_LIBM_SSE_EXTERN1( exp )
	DECLARE_LIBM_SSE_EXTERN1( exp2 )
	DECLARE_LIBM_SSE_EXTERN1( exp10 )
	DECLARE_LIBM_SSE_EXTERN1( expm1 )

	DECLARE_LIBM_SSE_EXTERN1( log )
	DECLARE_LIBM_SSE_EXTERN1( log2 )
	DECLARE_LIBM_SSE_EXTERN1( log10 )
	DECLARE_LIBM_SSE_EXTERN1( log1p )

	DECLARE_LIBM_SSE_EXTERN1( sin )
	DECLARE_LIBM_SSE_EXTERN1( cos )
	DECLARE_LIBM_SSE_EXTERN1( tan )
}

#endif


#ifdef LSIMD_USE_MATH_FUNCTIONS

namespace lsimd
{

	LSIMD_ENSURE_INLINE
	inline sse_f32pk cbrt( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(cbrt)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk cbrt( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(cbrt)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk pow( const sse_f32pk& x, const sse_f32pk& e )
	{
		return LSIMD_SSE_F(pow)(x.v, e.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk pow( const sse_f64pk& x, const sse_f64pk& e )
	{
		return LSIMD_SSE_D(pow)(x.v, e.v);
	}

#ifdef LSIMD_HAS_SSE_HYPOT

	LSIMD_ENSURE_INLINE
	inline sse_f32pk hypot( const sse_f32pk& x, const sse_f32pk& y )
	{
		return LSIMD_SSE_F(hypot)(x.v, y.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk hypot( const sse_f64pk& x, const sse_f64pk& y )
	{
		return LSIMD_SSE_D(hypot)(x.v, y.v);
	}

#endif

	LSIMD_ENSURE_INLINE
	inline sse_f32pk exp( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(exp)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk exp( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(exp)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk exp2( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(exp2)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk exp2( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(exp2)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk exp10( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(exp10)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk exp10( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(exp10)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk expm1( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(expm1)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk expm1( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(expm1)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk log( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(log)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk log( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(log)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk log2( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(log2)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk log2( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(log2)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk log10( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(log10)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk log10( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(log10)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk log1p( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(log1p)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk log1p( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(log1p)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk sin( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(sin)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk sin( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(sin)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk cos( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(cos)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk cos( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(cos)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk tan( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(tan)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk tan( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(tan)(x.v);
	}

#ifdef LSIMD_HAS_SSE_ANTI_TRIGONO

	LSIMD_ENSURE_INLINE
	inline sse_f32pk asin( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(asin)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk asin( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(asin)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk acos( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(acos)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk acos( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(acos)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk atan( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(atan)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk atan( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(atan)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk atan2( const sse_f32pk& x, const sse_f32pk& y )
	{
		return LSIMD_SSE_F(atan2)(x.v, y.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk atan2( const sse_f64pk& x, const sse_f64pk& y )
	{
		return LSIMD_SSE_D(atan2)(x.v, y.v);
	}

#endif


#ifdef LSIMD_HAS_SSE_HYPERBOLIC

	LSIMD_ENSURE_INLINE
	inline sse_f32pk sinh( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(sinh)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk sinh( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(sinh)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk cosh( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(cosh)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk cosh( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(cosh)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk tanh( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(tanh)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk tanh( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(tanh)(x.v);
	}
#endif


#ifdef LSIMD_HAS_SSE_ANTI_HYPERBOLIC

	LSIMD_ENSURE_INLINE
	inline sse_f32pk asinh( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(asinh)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk asinh( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(asinh)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk acosh( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(acosh)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk acosh( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(acosh)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk atanh( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(atanh)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk atanh( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(atanh)(x.v);
	}

#endif


#ifdef LSIMD_HAS_SSE_ERF

	LSIMD_ENSURE_INLINE
	inline sse_f32pk erf( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(erf)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk erf( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(erf)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk erfc( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(erfc)(x.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk erfc( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(erfc)(x.v);
	}

#endif /* LSIMD_HAS_SSE_ERF */

}

#undef LSIMD_SSE_F
#undef LSIMD_SSE_D

#endif /* LSIMD_USE_MATH_FUNCTIONS */



#ifdef LSIMD_USE_SVML

#undef DECLARE_SVML_SSE_EXTERN1

#undef SVML_SSE_F
#undef SVML_SSE_D

#endif  /* LSIMD_USE_SMVL */


#endif
