/*
 * @file simd_logical.h
 *
 * Comparison and logical operations on Generic SIMD packs.
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

#ifndef LSIMD_SIMD_LOGICAL_H_
#define LSIMD_SIMD_LOGICAL_H_

#include "simd_pack.h"
#include <light_simd/sse/sse_logical.h>

namespace lsimd
{
	/**
	 * @defgroup logical SIMD Comparison and Logicals
	 * @ingroup core_module
	 *
	 * @brief Overloaded comparison and logical operators
	 *
	 * This module provides operators and functions for
	 * comparison and logical operations, as well as
	 * conditional selection.
	 */

	/**
	 * @defgroup logical_generic Generic Comparison and Logicals
	 * @ingroup logical
	 *
	 * @brief Generic logical and comparison functions.
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
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator == (const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return a.impl == b.impl;
	}

	/**
	 * Entry-wise comparison for not-equal.
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a != b.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator != (const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return a.impl != b.impl;
	}

	/**
	 * Entry-wise comparison for less-than.
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a < b.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator < (const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return a.impl < b.impl;
	}

	/**
	 * Entry-wise comparison for less-than and equal-to
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a <= b.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator <= (const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return a.impl <= b.impl;
	}

	/**
	 * Entry-wise comparison for greater-than.
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a > b.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator > (const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return a.impl > b.impl;
	}

	/**
	 * Entry-wise comparison for greater-than and equal-to
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a >= b.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator >= (const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return a.impl >= b.impl;
	}


	/**
	 * bit-wise not.
	 *
	 * @param a  The input pack.
	 *
	 * @return   The resultant pack, as ~a.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator ~ (const simd_pack<T, Kind>& a)
	{
		return ~a.impl;
	}

	/**
	 * bit-wise and.
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a & b.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator & (const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return a.impl & b.impl;
	}

	/**
	 * bit-wise or.
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a | b.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator | (const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return a.impl | b.impl;
	}

	/**
	 * bit-wise xor.
	 *
	 * @param a  The left hand side pack.
	 * @param b  The right hand side pack.
	 *
	 * @return   The resultant pack, as a ^ b.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator ^ (const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return a.impl ^ b.impl;
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
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> cond(const simd_pack<T, Kind>& msk, const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return cond(msk.impl, a.impl, b.impl);
	}


	/** @} */  // logical_generic

}

#endif 
