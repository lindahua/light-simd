/**
 * @file sse_math.h
 *
 * @brief Transcendental math functions for SSE packs.
 *
 * @author Dahua Lin
 *
 * @copyright
 *
 * Copyright (C) 2012 Dahua Lin
 * 
 * Permission is hereby granted, free of charge, to any person 
 * obtaining a copy of this software and associated documentation 
 * files (the "Software"), to deal in the Software without restriction, 
 * including without limitation the rights to use, copy, modify, merge, 
 * publish, distribute, sublicense, and/or sell copies of the Software, 
 * and to permit persons to whom the Software is furnished to do so, 
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be 
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_MATH_H_
#define LSIMD_SSE_MATH_H_

#include "sse_pack.h"

#ifndef LSIMD_IN_DOXYGEN 

#if defined(LSIMD_USE_INTEL_SVML) || defined(LSIMD_USE_AMD_LIBM)
#define LSIMD_USE_MATH_FUNCTIONS
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
#define LIBM_SSE_F( name ) amd_vrd2_##name

#define DECLARE_LIBM_SSE_EXTERN1( name ) \
	__m128  SVML_SSE_F(name)( __m128 ); \
	__m128d SVML_SSE_D(name)( __m128d );

#define DECLARE_LIBM_SSE_EXTERN2( name ) \
	__m128  SVML_SSE_F(name)( __m128,  __m128  ); \
	__m128d SVML_SSE_D(name)( __m128d, __m128d );

#define LSIMD_SSE_F( name ) LIBM_SSE_F( name )
#define LSIMD_SSE_D( name ) LIBM_SSE_D( name )

extern "C"
{
	DECLARE_LIBM_SSE_EXTERN1( cbrt )
	DECLARE_LIBM_SSE_EXTERN2( pow )
	DECLARE_LIBM_SSE_EXTERN2( hypot )

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

	DECLARE_LIBM_SSE_EXTERN1( asin )
	DECLARE_LIBM_SSE_EXTERN1( acos )
	DECLARE_LIBM_SSE_EXTERN1( atan )
	DECLARE_LIBM_SSE_EXTERN2( atan2 )

	DECLARE_LIBM_SSE_EXTERN1( sinh )
	DECLARE_LIBM_SSE_EXTERN1( cosh )
	DECLARE_LIBM_SSE_EXTERN1( tanh )

	DECLARE_LIBM_SSE_EXTERN1( asinh )
	DECLARE_LIBM_SSE_EXTERN1( acosh )
	DECLARE_LIBM_SSE_EXTERN1( atanh )

	DECLARE_LIBM_SSE_EXTERN1( erf )
	DECLARE_LIBM_SSE_EXTERN1( erfc )
}

#endif

#endif // LSIMD_IN_DOXYGEN


#ifdef LSIMD_USE_MATH_FUNCTIONS

namespace lsimd
{

	/**
	 * @defgroup math_sse SSE Transcendental Math
	 * @ingroup math
	 *
	 * @brief Transcendental math functions on SSE packs.
	 */
	/** @{ */ 

	/**
	 * Evaluates cube roots in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as x^(1/3).
	 */
	LSIMD_ENSURE_INLINE sse_f32pk cbrt( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(cbrt)(x.v);
	}

	/**
	 * Evaluates cube roots in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as x^(1/3).
	 */
	LSIMD_ENSURE_INLINE sse_f64pk cbrt( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(cbrt)(x.v);
	}

	/**
	 * Evaluates powers in an entry-wise way.
	 *
	 * @param x  The pack of base values.
	 * @param e  The pack of exponents.
	 *
	 * @return   The resultant pack, as x^y.
	 */
	LSIMD_ENSURE_INLINE sse_f32pk pow( const sse_f32pk& x, const sse_f32pk& e )
	{
		return LSIMD_SSE_F(pow)(x.v, e.v);
	}

	/**
	 * Evaluates powers in an entry-wise way.
	 *
	 * @param x  The pack of base values.
	 * @param e  The pack of exponents.
	 *
	 * @return   The resultant pack, as x^y.
	 */
	LSIMD_ENSURE_INLINE sse_f64pk pow( const sse_f64pk& x, const sse_f64pk& e )
	{
		return LSIMD_SSE_D(pow)(x.v, e.v);
	}

    /**
	 * Evaluates hypotenuse values in an entry-wise way.
	 *
	 * @param x  The pack of x coordinates.
	 * @param y  The pack of y coordinates.
	 *
	 * @return   The resultant pack, as sqrt(x^2 + y^2).
	 */
	LSIMD_ENSURE_INLINE sse_f32pk hypot( const sse_f32pk& x, const sse_f32pk& y )
	{
		return LSIMD_SSE_F(hypot)(x.v, y.v);
	}

    /**
	 * Evaluates hypotenuse values in an entry-wise way.
	 *
	 * @param x  The pack of x coordinates.
	 * @param y  The pack of y coordinates.
	 *
	 * @return   The resultant pack, as sqrt(x^2 + y^2).
	 */
	LSIMD_ENSURE_INLINE sse_f64pk hypot( const sse_f64pk& x, const sse_f64pk& y )
	{
		return LSIMD_SSE_D(hypot)(x.v, y.v);
	}

	/**
	 * Evaluates exponentials in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as e^x.
	 */
	LSIMD_ENSURE_INLINE sse_f32pk exp( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(exp)(x.v);
	}

	/**
	 * Evaluates exponentials in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as e^x.
	 */
	LSIMD_ENSURE_INLINE sse_f64pk exp( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(exp)(x.v);
	}

	/**
	 * Evaluates exponentials (with base 2) in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as 2^x.
	 */
	LSIMD_ENSURE_INLINE sse_f32pk exp2( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(exp2)(x.v);
	}

	/**
	 * Evaluates exponentials (with base 2) in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as 2^x.
	 */
	LSIMD_ENSURE_INLINE sse_f64pk exp2( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(exp2)(x.v);
	}

	/**
	 * Evaluates exponentials (with base 10) in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as 10^x.
	 */
	LSIMD_ENSURE_INLINE sse_f32pk exp10( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(exp10)(x.v);
	}

	/**
	 * Evaluates exponentials (with base 10) in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as 10^x.
	 */
	LSIMD_ENSURE_INLINE sse_f64pk exp10( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(exp10)(x.v);
	}

	/**
	 * Evaluates the values of exponential minus 1 in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as e^x - 1.
	 */
	LSIMD_ENSURE_INLINE sse_f32pk expm1( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(expm1)(x.v);
	}

	/**
	 * Evaluates the values of exponential minus 1 in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as e^x - 1.
	 */
	LSIMD_ENSURE_INLINE sse_f64pk expm1( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(expm1)(x.v);
	}

	/**
	 * Evaluates natural logarithms in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as ln(x).
	 */
	LSIMD_ENSURE_INLINE sse_f32pk log( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(log)(x.v);
	}

	/**
	 * Evaluates natural logarithms in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as ln(x).
	 */
	LSIMD_ENSURE_INLINE sse_f64pk log( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(log)(x.v);
	}

	/**
	 * Evaluates logarithms (with base 2) in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as log_2(x).
	 */
	LSIMD_ENSURE_INLINE sse_f32pk log2( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(log2)(x.v);
	}

	/**
	 * Evaluates logarithms (with base 2) in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as log_2(x).
	 */
	LSIMD_ENSURE_INLINE sse_f64pk log2( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(log2)(x.v);
	}

	/**
	 * Evaluates logarithms (with base 10) in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as log_10(x).
	 */
	LSIMD_ENSURE_INLINE sse_f32pk log10( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(log10)(x.v);
	}

	/**
	 * Evaluates logarithms (with base 10) in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as log_10(x).
	 */
	LSIMD_ENSURE_INLINE sse_f64pk log10( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(log10)(x.v);
	}

	/**
	 * Evaluates the values of natural logarithms of (1 + x) in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as ln(1 + x).
	 */
	LSIMD_ENSURE_INLINE sse_f32pk log1p( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(log1p)(x.v);
	}

	/**
	 * Evaluates the values of natural logarithms of (1 + x) in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as ln(1 + x).
	 */
	LSIMD_ENSURE_INLINE sse_f64pk log1p( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(log1p)(x.v);
	}

	/**
	 * Evaluates sine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as sin(x).
	 */
	LSIMD_ENSURE_INLINE sse_f32pk sin( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(sin)(x.v);
	}

	/**
	 * Evaluates sine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as sin(x).
	 */
	LSIMD_ENSURE_INLINE sse_f64pk sin( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(sin)(x.v);
	}

	/**
	 * Evaluates cosine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as cos(x).
	 */
	LSIMD_ENSURE_INLINE sse_f32pk cos( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(cos)(x.v);
	}

	/**
	 * Evaluates cosine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as cos(x).
	 */
	LSIMD_ENSURE_INLINE sse_f64pk cos( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(cos)(x.v);
	}

	/**
	 * Evaluates tangent values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as sin(x).
	 */
	LSIMD_ENSURE_INLINE sse_f32pk tan( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(tan)(x.v);
	}

	/**
	 * Evaluates tangent values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as sin(x).
	 */
	LSIMD_ENSURE_INLINE sse_f64pk tan( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(tan)(x.v);
	}

	/**
	 * Evaluates arc sine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as arcsin(x).
	 */
	LSIMD_ENSURE_INLINE sse_f32pk asin( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(asin)(x.v);
	}

	/**
	 * Evaluates arc sine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as arcsin(x).
	 */
	LSIMD_ENSURE_INLINE sse_f64pk asin( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(asin)(x.v);
	}

	/**
	 * Evaluates arc cosine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as arccos(x).
	 */
	LSIMD_ENSURE_INLINE sse_f32pk acos( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(acos)(x.v);
	}

	/**
	 * Evaluates arc cosine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as arccos(x).
	 */
	LSIMD_ENSURE_INLINE sse_f64pk acos( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(acos)(x.v);
	}

	/**
	 * Evaluates arc tangent values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as arctan(x).
	 */
	LSIMD_ENSURE_INLINE sse_f32pk atan( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(atan)(x.v);
	}

	/**
	 * Evaluates arc tangent values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as arctan(x).
	 */
	LSIMD_ENSURE_INLINE sse_f64pk atan( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(atan)(x.v);
	}

	/**
	 * Evaluates arc tangent values of y / x in an entry-wise way.
	 *
	 * @param x  The input pack of x.
	 * @param y  The input pack of y.
	 *
	 * @return   The resultant pack, as arctan(y/x).
	 */
	LSIMD_ENSURE_INLINE sse_f32pk atan2( const sse_f32pk& x, const sse_f32pk& y )
	{
		return LSIMD_SSE_F(atan2)(x.v, y.v);
	}

	/**
	 * Evaluates arc tangent values of y / x in an entry-wise way.
	 *
	 * @param x  The input pack of x.
	 * @param y  The input pack of y.
	 *
	 * @return   The resultant pack, as arctan(y/x).
	 */
	LSIMD_ENSURE_INLINE sse_f64pk atan2( const sse_f64pk& x, const sse_f64pk& y )
	{
		return LSIMD_SSE_D(atan2)(x.v, y.v);
	}


	/**
	 * Evaluates hyperbolic sine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as sinh(x).
	 */
	LSIMD_ENSURE_INLINE sse_f32pk sinh( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(sinh)(x.v);
	}

	/**
	 * Evaluates hyperbolic sine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as sinh(x).
	 */
	LSIMD_ENSURE_INLINE sse_f64pk sinh( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(sinh)(x.v);
	}

	/**
	 * Evaluates hyperbolic cosine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as sinh(x).
	 */
	LSIMD_ENSURE_INLINE sse_f32pk cosh( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(cosh)(x.v);
	}

	/**
	 * Evaluates hyperbolic cosine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as sinh(x).
	 */
	LSIMD_ENSURE_INLINE sse_f64pk cosh( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(cosh)(x.v);
	}

	/**
	 * Evaluates hyperbolic tangent values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as tanh(x).
	 */
	LSIMD_ENSURE_INLINE sse_f32pk tanh( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(tanh)(x.v);
	}

	/**
	 * Evaluates hyperbolic tangent values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as tanh(x).
	 */
	LSIMD_ENSURE_INLINE sse_f64pk tanh( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(tanh)(x.v);
	}


	/**
	 * Evaluates hyperbolic arc sine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack.
	 */
	LSIMD_ENSURE_INLINE sse_f32pk asinh( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(asinh)(x.v);
	}

	/**
	 * Evaluates hyperbolic arc sine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack.
	 */
	LSIMD_ENSURE_INLINE sse_f64pk asinh( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(asinh)(x.v);
	}

	/**
	 * Evaluates hyperbolic arc cosine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack.
	 */
	LSIMD_ENSURE_INLINE sse_f32pk acosh( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(acosh)(x.v);
	}

	/**
	 * Evaluates hyperbolic arc cosine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack.
	 */
	LSIMD_ENSURE_INLINE sse_f64pk acosh( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(acosh)(x.v);
	}

	/**
	 * Evaluates hyperbolic arc tangent values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack.
	 */
	LSIMD_ENSURE_INLINE sse_f32pk atanh( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(atanh)(x.v);
	}

	/**
	 * Evaluates hyperbolic arc tangent values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack.
	 */
	LSIMD_ENSURE_INLINE sse_f64pk atanh( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(atanh)(x.v);
	}

#ifdef LSIMD_HAS_SSE_ERF

	/**
	 * Evaluates the error function values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack.
	 */
	LSIMD_ENSURE_INLINE sse_f32pk erf( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(erf)(x.v);
	}

	/**
	 * Evaluates the error function values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack.
	 */
	LSIMD_ENSURE_INLINE sse_f64pk erf( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(erf)(x.v);
	}

	/**
	 * Evaluates the complementary error function values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack.
	 */
	LSIMD_ENSURE_INLINE sse_f32pk erfc( const sse_f32pk& x )
	{
		return LSIMD_SSE_F(erfc)(x.v);
	}

	/**
	 * Evaluates the complementary error function values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack.
	 */
	LSIMD_ENSURE_INLINE sse_f64pk erfc( const sse_f64pk& x )
	{
		return LSIMD_SSE_D(erfc)(x.v);
	}

#endif /* LSIMD_HAS_SSE_ERF */

	/** @} */
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
