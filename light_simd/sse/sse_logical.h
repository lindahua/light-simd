/*
 * @file sse_logical.h
 *
 * @brief SSE Comparison and logical operators on SSE Packs.
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

#ifndef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_LOGICAL_H_
#define LSIMD_SSE_LOGICAL_H_

#include "sse_pack.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4141)
#endif

namespace lsimd
{
	/**
	 * @defgroup logical_sse SSE Comparison and Logicals
	 * @ingroup logical
	 *
	 * @brief SSE-based comparison and logical functions.
	 */
	/** @{ */


	/**
	 * Entry-wise comparison for equal.
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a == b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator == (const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_cmpeq_ps(a.v, b.v);
	}

	/**
	 * Entry-wise comparison for equal.
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a == b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator == (const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_cmpeq_pd(a.v, b.v);
	}

	/**
	 * Entry-wise comparison for not-equal.
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a != b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator != (const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_cmpneq_ps(a.v, b.v);
	}

	/**
	 * Entry-wise comparison for not-equal.
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a != b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator != (const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_cmpneq_pd(a.v, b.v);
	}


	/**
	 * Entry-wise comparison for less-than.
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a < b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator < (const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_cmplt_ps(a.v, b.v);
	}

	/**
	 * Entry-wise comparison for less-than.
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a < b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator < (const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_cmplt_pd(a.v, b.v);
	}

	/**
	 * Entry-wise comparison for less-than or equal-to
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a <= b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator <= (const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_cmple_ps(a.v, b.v);
	}

	/**
	 * Entry-wise comparison for less-than or equal-to
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a <= b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator <= (const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_cmple_pd(a.v, b.v);
	}

	/**
	 * Entry-wise comparison for greater-than.
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a > b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator > (const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_cmpgt_ps(a.v, b.v);
	}

	/**
	 * Entry-wise comparison for greater-than.
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a > b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator > (const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_cmpgt_pd(a.v, b.v);
	}

	/**
	 * Entry-wise comparison for greater-than or equal-to
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a >= b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator >= (const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_cmpge_ps(a.v, b.v);
	}

	/**
	 * Entry-wise comparison for greater-than or equal-to
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a >= b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator >= (const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_cmpge_pd(a.v, b.v);
	}


	/**
	 * bit-wise not.
	 *
	 * @param a  The input pack.
	 *
	 * @return   The resultant pack, as ~a.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator ~ (const sse_f32pk& a)
	{
		return _mm_xor_ps(a.v, _mm_castsi128_ps(sse::all_one_bits()));
	}

	/**
	 * bit-wise not.
	 *
	 * @param a  The input pack.
	 *
	 * @return   The resultant pack, as ~a.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator ~ (const sse_f64pk& a)
	{
		return _mm_xor_pd(a.v, _mm_castsi128_pd(sse::all_one_bits()));
	}

	/**
	 * bit-wise and.
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a & b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator & (const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_and_ps(a.v, b.v);
	}

	/**
	 * bit-wise and.
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a & b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator & (const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_and_pd(a.v, b.v);
	}


	/**
	 * bit-wise or.
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a | b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator | (const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_or_ps(a.v, b.v);
	}

	/**
	 * bit-wise or.
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a | b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator | (const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_or_pd(a.v, b.v);
	}


	/**
	 * bit-wise xor.
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a ^ b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator ^ (const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_xor_ps(a.v, b.v);
	}

	/**
	 * bit-wise xor.
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a ^ b.
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator ^ (const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_xor_pd(a.v, b.v);
	}

#ifndef LSIMD_IN_DOXYGEN

	LSIMD_ENSURE_INLINE
	inline sse_f32pk cond_sse2(const sse_f32pk& msk, const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_or_ps(_mm_and_ps(msk.v, a.v), _mm_andnot_ps(msk.v, b.v));
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk cond_sse2(const sse_f64pk& msk, const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_or_pd(_mm_and_pd(msk.v, a.v), _mm_andnot_pd(msk.v, b.v));
	}

#ifdef LSIMD_HAS_SSE4_1
	LSIMD_ENSURE_INLINE
	inline sse_f32pk cond_sse4(const sse_f32pk& msk, const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_blendv_ps(b.v, a.v, msk.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk cond_sse4(const sse_f64pk& msk, const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_blendv_pd(b.v, a.v, msk.v);
	}
#endif

#endif


	/**
	 * Entry-wise conditional selection.
	 *
	 * @param msk 	The selection mask
	 * @param a		The pack of values used when mask-values are true.
	 * @param b		The pack of values used when mask-values are false.
	 *
	 * @return		The resultant pack, as (msk ? a : b).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk cond(const sse_f32pk& msk, const sse_f32pk& a, const sse_f32pk& b)
	{
#ifdef LSIMD_HAS_SSE4_1
		return cond_sse4(msk, a, b);
#else
		return cond_sse2(msk, a, b);
#endif
	}

	/**
	 * Entry-wise conditional selection.
	 *
	 * @param msk 	The selection mask
	 * @param a		The pack of values used when mask-values are true.
	 * @param b		The pack of values used when mask-values are false.
	 *
	 * @return		The resultant pack, as (msk ? a : b).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk cond(const sse_f64pk& msk, const sse_f64pk& a, const sse_f64pk& b)
	{
#ifdef LSIMD_HAS_SSE4_1
		return cond_sse4(msk, a, b);
#else
		return cond_sse2(msk, a, b);
#endif
	}


	/** @} */  // logical_sse
}


#endif 





