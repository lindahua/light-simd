/**
 * @file simd_math.h
 *
 * @brief Transcendental math functions for Generic SIMD packs.
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SIMD_MATH_H_
#define LSIMD_SIMD_MATH_H_

#include "simd_pack.h"

#include <light_simd/sse/sse_math.h>

#define LSIMD_DEFINE_SIMD_MATH_FUNC1(fun) \
	template<typename T, typename Kind> \
	LSIMD_ENSURE_INLINE \
	inline simd_pack<T, Kind> fun(const simd_pack<T, Kind>& x) \
	{ return fun(x.impl); }


#define LSIMD_DEFINE_SIMD_MATH_FUNC2(fun) \
	template<typename T, typename Kind> \
	LSIMD_ENSURE_INLINE \
	inline simd_pack<T, Kind> fun(const simd_pack<T, Kind>& x, const simd_pack<T, Kind>& y) \
	{ return fun(x.impl, y.impl); }


namespace lsimd
{

	LSIMD_DEFINE_SIMD_MATH_FUNC1( cbrt )
	LSIMD_DEFINE_SIMD_MATH_FUNC2( pow )
	LSIMD_DEFINE_SIMD_MATH_FUNC2( hypot )

	LSIMD_DEFINE_SIMD_MATH_FUNC1( exp )
	LSIMD_DEFINE_SIMD_MATH_FUNC1( exp2 )
	LSIMD_DEFINE_SIMD_MATH_FUNC1( exp10 )
	LSIMD_DEFINE_SIMD_MATH_FUNC1( expm1 )

	LSIMD_DEFINE_SIMD_MATH_FUNC1( log )
	LSIMD_DEFINE_SIMD_MATH_FUNC1( log2 )
	LSIMD_DEFINE_SIMD_MATH_FUNC1( log10 )
	LSIMD_DEFINE_SIMD_MATH_FUNC1( log1p )

	LSIMD_DEFINE_SIMD_MATH_FUNC1( sin )
	LSIMD_DEFINE_SIMD_MATH_FUNC1( cos )
	LSIMD_DEFINE_SIMD_MATH_FUNC1( tan )

	LSIMD_DEFINE_SIMD_MATH_FUNC1( asin )
	LSIMD_DEFINE_SIMD_MATH_FUNC1( acos )
	LSIMD_DEFINE_SIMD_MATH_FUNC1( atan )
	LSIMD_DEFINE_SIMD_MATH_FUNC2( atan2 )

	LSIMD_DEFINE_SIMD_MATH_FUNC1( sinh )
	LSIMD_DEFINE_SIMD_MATH_FUNC1( cosh )
	LSIMD_DEFINE_SIMD_MATH_FUNC1( tanh )

	LSIMD_DEFINE_SIMD_MATH_FUNC1( asinh )
	LSIMD_DEFINE_SIMD_MATH_FUNC1( acosh )
	LSIMD_DEFINE_SIMD_MATH_FUNC1( atanh )

#ifdef LSIMD_HAS_SSE_ERF
	LSIMD_DEFINE_SIMD_MATH_FUNC1( erf )
	LSIMD_DEFINE_SIMD_MATH_FUNC1( erfc )
#endif
}


#undef LSIMD_DEFINE_SIMD_MATH_FUNC1
#undef LSIMD_DEFINE_SIMD_MATH_FUNC2


#endif



