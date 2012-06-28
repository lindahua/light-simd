/**
 * @file sse_arith.h
 *
 * @brief Arithmetic operators and functions for SSE packs.
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

#ifndef LSIMD_SSE_ARITH_H_
#define LSIMD_SSE_ARITH_H_

#include "sse_pack.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4141)
#endif

namespace lsimd
{
	/**
	 * @defgroup arith_sse SSE Arithmetics
	 * @ingroup arith
	 *
	 * @brief SSE-based arithmetic operators and functions.
	 */ 
	/** @{ */ 

	/**
	 * Adds two packs in an entry-wise way.
	 *
	 * @param a  The pack of summands.
	 * @param b  The pack of addends.
	 *
	 * @return   The resultant pack, as a + b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator + (const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_add_ps(a.v, b.v);
	}

	/**
	 * Adds two packs in an entry-wise way.
	 *
	 * @param a  The pack of summands.
	 * @param b  The pack of addends.
	 *
	 * @return   The resultant pack, as a + b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator + (const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_add_pd(a.v, b.v);
	}

	/**
	 * Subtracts a pack b from another pack a in an entry-wise way.
	 *
	 * @param a  The pack of minuends.
	 * @param b  The pack of subtrahends.
	 *
	 * @return   The resultant pack, as a - b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator - (const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_sub_ps(a.v, b.v);
	}

	/**
	 * Subtracts a pack b from another pack a in an entry-wise way.
	 *
	 * @param a  The pack of minuends.
	 * @param b  The pack of subtrahends.
	 *
	 * @return   The resultant pack, as a - b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator - (const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_sub_pd(a.v, b.v);
	}

	/**
	 * Multiplies two packs in an entry-wise way.
	 *
	 * @param a  The pack of multiplicands.
	 * @param b  The pack of multipliers.
	 *
	 * @return   The resultant pack, as a * b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator * (const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_mul_ps(a.v, b.v);
	}

	/**
	 * Multiplies two packs in an entry-wise way.
	 *
	 * @param a  The pack of multiplicands.
	 * @param b  The pack of multipliers.
	 *
	 * @return   The resultant pack, as a * b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator * (const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_mul_pd(a.v, b.v);
	}

	/**
	 * Divides a pack b from another pack a in an entry-wise way.
	 *
	 * @param a  The pack of dividends
	 * @param b  The pack of divisors
	 *
	 * @return   The resultant pack, as a / b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator / (const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_div_ps(a.v, b.v);
	}

	/**
	 * Divides a pack b from another pack a in an entry-wise way.
	 *
	 * @param a  The pack of dividends
	 * @param b  The pack of divisors
	 *
	 * @return   The resultant pack, as a / b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator / (const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_div_pd(a.v, b.v);
	}

	/**
	 * Negates a pack in an entry-wise way.
	 *
	 * @param a  The input pack.
	 *
	 * @return   The resultant pack, as -a.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator - (const sse_f32pk& a)
	{
		return _mm_xor_ps(_mm_set1_ps(-0.f), a.v);
	}

	/**
	 * Negates a pack in an entry-wise way.
	 *
	 * @param a  The input pack.
	 *
	 * @return   The resultant pack, as -a.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator - (const sse_f64pk& a)
	{
		return _mm_xor_pd(_mm_set1_pd(-0.0), a.v);
	}

	/**
	 * Evaluates the absolute values in an entry-wise way.
	 *
	 * @param a  The input pack.
	 *
	 * @return   The resultant pack, as |a|.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk abs(const sse_f32pk& a)
	{
		return _mm_andnot_ps(_mm_set1_ps(-0.f), a.v);
	}

	/**
	 * Evaluates the absolute values in an entry-wise way.
	 *
	 * @param a  The input pack.
	 *
	 * @return   The resultant pack, as |a|.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk abs(const sse_f64pk& a)
	{
		return _mm_andnot_pd(_mm_set1_pd(-0.0), a.v);
	}

	/**
	 * Selects the smaller values between two packs in an entry-wise way.
	 *
	 * @param a   The first input pack.
	 * @param b   The second input pack.
	 *
	 * @return    The resultant pack.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk vmin(const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_min_ps(a.v, b.v);
	}

	/**
	 * Selects the smaller values between two packs in an entry-wise way.
	 *
	 * @param a   The first input pack.
	 * @param b   The second input pack.
	 *
	 * @return    The resultant pack.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk vmin(const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_min_pd(a.v, b.v);
	}

	/**
	 * Selects the larger values between two packs in an entry-wise way.
	 *
	 * @param a   The first input pack.
	 * @param b   The second input pack.
	 *
	 * @return    The resultant pack.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk vmax(const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_max_ps(a.v, b.v);
	}

	/**
	 * Selects the larger values between two packs in an entry-wise way.
	 *
	 * @param a   The first input pack.
	 * @param b   The second input pack.
	 *
	 * @return    The resultant pack.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk vmax(const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_max_pd(a.v, b.v);
	}


	/**
	 * Calculates the squared roots in an entry-wise way.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as sqrt(a), i.e. a^(1/2).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk sqrt(const sse_f32pk& a)
	{
		return _mm_sqrt_ps(a.v);
	}

	/**
	 * Calculates the squared roots in an entry-wise way.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as sqrt(a), i.e. a^(1/2).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk sqrt(const sse_f64pk& a)
	{
		return _mm_sqrt_pd(a.v);
	}

	/**
	 * Calculates the reciprocals in an entry-wise way.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as 1 / a.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk rcp(const sse_f32pk& a)
	{
		return _mm_div_ps(_mm_set1_ps(1.0f), a.v);
	}

	/**
	 * Calculates the approximate reciprocals in an entry-wise way.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as 1 / a.
	 *
	 * @remark    This function invokes the RCPPS instruction, which
	 *            is very fast, but only yielding approximate results.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk approx_rcp(const sse_f32pk& a)
	{
		return _mm_rcp_ps(a.v);
	}

	/**
	 * Calculates the reciprocals in an entry-wise way.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as 1 / a.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk rcp(const sse_f64pk& a)
	{
		return _mm_div_pd(_mm_set1_pd(1.0), a.v);
	}

	/**
	 * Calculates the reciprocals of the squared-roots in 
	 * an entry-wise way.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as 1 / sqrt(a), i.e. a^(-1/2).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk rsqrt(const sse_f32pk& a)
	{
		return _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(a.v));
	}

	/**
	 * Calculates the reciprocals of the squared-roots in 
	 * an entry-wise way.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as 1 / sqrt(a), i.e. a^(-1/2).
	 *
	 * @remark    This function invokes the RSQRTPS instruction, which
	 *            is very fast, but only yielding approximate results.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk approx_rsqrt(const sse_f32pk& a)
	{
		return _mm_rsqrt_ps(a.v);
	}

	/**
	 * Calculates the reciprocals of the squared-roots in 
	 * an entry-wise way.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as 1 / sqrt(a), i.e. a^(-1/2).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk rsqrt(const sse_f64pk& a)
	{
		return _mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(a.v));
	}

	/**
	 * Calculates the squares in an entry-wise way.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as a^2.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk sqr(const sse_f32pk& a)
	{
		return _mm_mul_ps(a.v, a.v);
	}

	/**
	 * Calculates the squares in an entry-wise way.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as a^2.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk sqr(const sse_f64pk& a)
	{
		return _mm_mul_pd(a.v, a.v);
	}

	/**
	 * Calculates the cubes in an entry-wise way.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as a^3.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk cube(const sse_f32pk& a)
	{
		return _mm_mul_ps(_mm_mul_ps(a.v, a.v), a.v);
	}

	/**
	 * Calculates the cubes in an entry-wise way.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as a^3.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk cube(const sse_f64pk& a)
	{
		return _mm_mul_pd(_mm_mul_pd(a.v, a.v), a.v);
	}


	/********************************************
	 *
	 *  Single-scalar arithmetic
	 *
	 ********************************************/

	/**
	 * Adds the first entry of b to that of a.
	 *
	 * @param a  The pack of summands.
	 * @param b  The pack of addends.
	 *
	 * @return   The resultant pack, 
	 *           as (a[0] + b[0], a[1], a[2], a[3]).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk add_s(const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_add_ss(a.v, b.v);
	}

	/**
	 * Adds the first entry of b to that of a.
	 *
	 * @param a  The pack of summands.
	 * @param b  The pack of addends.
	 *
	 * @return   The resultant pack, 
	 *           as (a[0] + b[0], a[1]).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk add_s(const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_add_sd(a.v, b.v);
	}

	/**
	 * Subtracts the first entry of b from that of a.
	 *
	 * @param a  The pack of minuends.
	 * @param b  The pack of subtrahends.
	 *
	 * @return   The resultant pack, 
	 *           as (a[0] - b[0], a[1], a[2], a[3]).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk sub_s(const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_sub_ss(a.v, b.v);
	}

	/**
	 * Subtracts the first entry of b from that of a.
	 *
	 * @param a  The pack of minuends.
	 * @param b  The pack of subtrahends.
	 *
	 * @return   The resultant pack, 
	 *           as (a[0] - b[0], a[1]).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk sub_s(const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_sub_sd(a.v, b.v);
	}

	/**
	 * Multiplies the first entry of b to that of a.
	 *
	 * @param a  The pack of multiplicands.
	 * @param b  The pack of multipliers.
	 *
	 * @return   The resultant pack, 
	 *           as (a[0] * b[0], a[1], a[2], a[3]).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk mul_s(const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_mul_ss(a.v, b.v);
	}

	/**
	 * Multiplies the first entry of b to that of a.
	 *
	 * @param a  The pack of multiplicands.
	 * @param b  The pack of multipliers.
	 *
	 * @return   The resultant pack, 
	 *           as (a[0] * b[0], a[1]).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk mul_s(const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_mul_sd(a.v, b.v);
	}

	/**
	 * Divides the first entry of b from that of a.
	 *
	 * @param a  The pack of dividends
	 * @param b  The pack of divisors.
	 *
	 * @return   The resultant pack, 
	 *           as (a[0] / b[0], a[1], a[2], a[3]).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk div_s(const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_div_ss(a.v, b.v);
	}

	/**
	 * Divides the first entry of b from that of a.
	 *
	 * @param a  The pack of dividends
	 * @param b  The pack of divisors.
	 *
	 * @return   The resultant pack, 
	 *           as (a[0] / b[0], a[1]).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk div_s(const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_div_sd(a.v, b.v);
	}

	/**
	 * Calculates the reciprocal of the first entry.
	 *
	 * @param a  The input pack
	 *
	 * @return   The resultant pack, as (1 / a[0], 0, 0, 0).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk rcp_s(const sse_f32pk& a)
	{
		return _mm_div_ss(_mm_set_ss(1.0f), a.v);
	}

	/**
	 * Calculates the reciprocal of the first entry.
	 *
	 * @param a  The input pack
	 *
	 * @return   The resultant pack, as (1 / a[0], 0).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk rcp_s(const sse_f64pk& a)
	{
		return _mm_div_sd(_mm_set_sd(1.0), a.v);
	}

	/**
	 * Calculates the squared root of the first entry.
	 *
	 * @param a  The input pack
	 *
	 * @return   The resultant pack, as (sqrt(a[0]), 0, 0, 0).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk sqrt_s(const sse_f32pk& a)
	{
		return _mm_sqrt_ss(a.v);
	}

	/**
	 * Calculates the squared root of the first entry.
	 *
	 * @param a  The input pack
	 *
	 * @return   The resultant pack, as (sqrt(a[0]), 0).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk sqrt_s(const sse_f64pk& a)
	{
		return _mm_sqrt_sd(a.v, a.v);
	}



#ifndef LSIMD_IN_DOXYGEN

	LSIMD_ENSURE_INLINE
	inline sse_f32pk floor_sse2(const sse_f32pk& a)
	{
		__m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(a.v));
		__m128 b = _mm_and_ps(_mm_cmpgt_ps(t, a.v), _mm_set1_ps(1.0f));

		return _mm_sub_ps(t, b);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk floor_sse2(const sse_f64pk& a)
	{
		__m128d t = _mm_cvtepi32_pd(_mm_cvttpd_epi32(a.v));
		__m128d b = _mm_and_pd(_mm_cmpgt_pd(t, a.v), _mm_set1_pd(1.0));

		return _mm_sub_pd(t, b);
	}


	LSIMD_ENSURE_INLINE
	inline sse_f32pk ceil_sse2(const sse_f32pk& a)
	{
		__m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(a.v));
		__m128 b = _mm_and_ps(_mm_cmplt_ps(t, a.v), _mm_set1_ps(1.0f));

		return _mm_add_ps(t, b);
	}


	LSIMD_ENSURE_INLINE
	inline sse_f64pk ceil_sse2(const sse_f64pk& a)
	{
		__m128d t = _mm_cvtepi32_pd(_mm_cvttpd_epi32(a.v));
		__m128d b = _mm_and_pd(_mm_cmplt_pd(t, a.v), _mm_set1_pd(1.0));

		return _mm_add_pd(t, b);
	}


#ifdef LSIMD_HAS_SSE4_1
	LSIMD_ENSURE_INLINE
	inline sse_f32pk floor_sse4(const sse_f32pk& a)
	{
		return _mm_floor_ps(a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk floor_sse4(const sse_f64pk& a)
	{
		return _mm_floor_pd(a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk ceil_sse4(const sse_f32pk& a)
	{
		return _mm_ceil_ps(a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk ceil_sse4(const sse_f64pk& a)
	{
		return _mm_ceil_pd(a.v);
	}
#endif

#endif /* LSIMD_IN_DOXYGEN */


	/**
	 * Calculates the floor values in an entry-wise way.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as floor(a).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk floor(const sse_f32pk& a)
	{
#ifdef LSIMD_HAS_SSE4_1
		return floor_sse4(a);
#else
		return floor_sse2(a);
#endif
	}

	/**
	 * Calculates the floor values in an entry-wise way.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as floor(a).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk floor(const sse_f64pk& a)
	{
#ifdef LSIMD_HAS_SSE4_1
		return floor_sse4(a);
#else
		return floor_sse2(a);
#endif
	}

	/**
	 * Calculates the ceil values in an entry-wise way.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as ceil(a).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk ceil(const sse_f32pk& a)
	{
#ifdef LSIMD_HAS_SSE4_1
		return ceil_sse4(a);
#else
		return ceil_sse2(a);
#endif
	}

	/**
	 * Calculates the ceil values in an entry-wise way.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as ceil(a).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk ceil(const sse_f64pk& a)
	{
#ifdef LSIMD_HAS_SSE4_1
		return ceil_sse4(a);
#else
		return ceil_sse2(a);
#endif
	}

	/** @} */ // arith_sse

}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif /* SSE_ARITH_H_ */
