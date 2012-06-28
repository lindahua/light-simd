/**
 * @file simd_math.h
 *
 * @brief Transcendental math functions for Generic SIMD packs.
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

#ifndef LSIMD_SIMD_MATH_H_
#define LSIMD_SIMD_MATH_H_

#include "simd_pack.h"

#include <light_simd/sse/sse_math.h>

#define LSIMD_DEFINE_SIMD_MATH_FUNC1(fun) \
	template<typename T, typename Kind> \
	LSIMD_ENSURE_INLINE \
	simd_pack<T, Kind> fun(const simd_pack<T, Kind>& x) \
	{ return fun(x.impl); }


#define LSIMD_DEFINE_SIMD_MATH_FUNC2(fun) \
	template<typename T, typename Kind> \
	LSIMD_ENSURE_INLINE \
	simd_pack<T, Kind> fun(const simd_pack<T, Kind>& x, const simd_pack<T, Kind>& y) \
	{ return fun(x.impl, y.impl); }


namespace lsimd
{
	/**
	 * @defgroup math Math Module
	 * 
	 * @brief Math functions on SIMD packs 
	 *
	 * This module provides a set of transcendental math functions, including
	 * - The set of math functions covered by *C89* or *C++03* standards:
	 *    - Power function: *pow*
	 *    - Exponential and logarithmic functions: *exp, log, log10*
	 *    - Trigonometric functions: *sin, cos, tan, acos, asin, atan, atan2*
	 *    - Hyperbolic functions: *sinh, cosh, tanh*
	 *
	 * - The extended set of math functions covered by *C99* or *C++11* standards:
	 *    - Power functions: *cbrt, *hypot*
	 *    - Exponential and logarithmic functions: *exp2, exp10, log2, expm1, log1p*
	 *    - Hyperbolic functions: *asinh, acosh, atanh*
	 *    - Normal error functions: *erf, erfc*
	 *
	 * In current version, we delegate the actual implementation to either 
	 * [Intel SVML (Short Vector Math Library)]
	 * (http://software.intel.com/sites/products/documentation/hpc/composerxe/en-us/cpp/lin/intref_cls/common/intref_svml_overview.htm) 
	 * or [AMD LibM]
	 * (http://developer.amd.com/libraries/LibM).
	 * 
	 * @remark  Some basic functions like *sqrt*, *floor* and *ceil* are in the 
	 *          \ref arith module, as we don't rely on SVML or LibM to provide 
	 *          such functions.
	 */

	/**
	 * @defgroup math_generic Generic Transcendental Math
	 * @ingroup math
	 *
	 * @brief Transcendental math functions on generic SIMD packs.
	 */
	/** @{ */ 

	/**
	 * Evaluates cube roots in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as x^(1/3).
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( cbrt )

	/**
	 * Evaluates powers in an entry-wise way.
	 *
	 * @param x  The pack of base values.
	 * @param y  The pack of exponents.
	 *
	 * @return   The resultant pack, as x^y.
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC2( pow )

    /**
	 * Evaluates hypotenuse values in an entry-wise way.
	 *
	 * @param x  The pack of x coordinates.
	 * @param y  The pack of y coordinates.
	 *
	 * @return   The resultant pack, as sqrt(x^2 + y^2).
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC2( hypot )

	/**
	 * Evaluates exponentials in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as e^x.
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( exp )

	/**
	 * Evaluates exponentials (with base 2) in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as 2^x.
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( exp2 )

	/**
	 * Evaluates exponentials (with base 10) in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as 10^x.
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( exp10 )

	/**
	 * Evaluates the values of exponential minus 1 in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as e^x - 1.
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( expm1 )

	/**
	 * Evaluates natural logarithms in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as ln(x).
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( log )

	/**
	 * Evaluates logarithms (with base 2) in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as log_2(x).
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( log2 )

	/**
	 * Evaluates logarithms (with base 10) in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as log_10(x).
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( log10 )

	/**
	 * Evaluates the values of natural logarithms of (1 + x) in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as ln(1 + x).
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( log1p )

	/**
	 * Evaluates sine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as sin(x).
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( sin )

	/**
	 * Evaluates cosine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as cos(x).
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( cos )

	/**
	 * Evaluates tangent values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as sin(x).
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( tan )

	/**
	 * Evaluates arc sine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as arcsin(x).
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( asin )

	/**
	 * Evaluates arc cosine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as arccos(x).
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( acos )

	/**
	 * Evaluates arc tangent values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as arctan(x).
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( atan )

	/**
	 * Evaluates arc tangent values of y / x in an entry-wise way.
	 *
	 * @param x  The input pack of x.
	 * @param y  The input pack of y.
	 *
	 * @return   The resultant pack, as arctan(y/x).
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC2( atan2 )

	/**
	 * Evaluates hyperbolic sine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as sinh(x).
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( sinh )

	/**
	 * Evaluates hyperbolic cosine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as sinh(x).
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( cosh )

	/**
	 * Evaluates hyperbolic tangent values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack, as tanh(x).
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( tanh )

	/**
	 * Evaluates hyperbolic arc sine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack.
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( asinh )

	/**
	 * Evaluates hyperbolic arc cosine values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack.
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( acosh )

	/**
	 * Evaluates hyperbolic arc tangent values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack.
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( atanh )

	/**
	 * Evaluates the error function values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack.
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( erf )

	/**
	 * Evaluates the complementary error function values in an entry-wise way.
	 *
	 * @param x  The input pack.
	 *
	 * @return   The resultant pack.
	 */
	LSIMD_DEFINE_SIMD_MATH_FUNC1( erfc )

	/** @} */

}


#undef LSIMD_DEFINE_SIMD_MATH_FUNC1
#undef LSIMD_DEFINE_SIMD_MATH_FUNC2


#endif /* SIMD_MATH_H_ */
