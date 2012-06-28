/**
 * @file simd_arith.h
 *
 * @brief Arithmetic operators and functions for Generic SIMD packs.
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

#ifndef LSIMD_SIMD_ARITH_H_
#define LSIMD_SIMD_ARITH_H_

#include "simd_pack.h"
#include <light_simd/sse/sse_arith.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4141)
#endif

namespace lsimd
{
	/**
	 * @defgroup arith SIMD Arithmetics
	 * @ingroup core_module
	 * 
	 * @brief Overloaded arithmetic operators and basic arithmetic 
	 *        functions (e.g. abs and sqrt).
	 *
	 * This module provides arithmetic operators and functions that 
	 * can be applied to both generic and architecture-specific 
	 * SIMD pack classes. 
	 */

	/**
	 * @defgroup arith_generic Generic Arithmetics
	 * @ingroup arith
	 *
	 * @brief Generic arithmetic operators and functions.
	 */ 
	/** @{ */ 

	/**
	 * Adds two packs in an entry-wise way.
	 *
	 * @tparam   The scalar type of the packs.
	 * @tparam   The SIMD kind of the packs.
	 * 
	 * @param a  The pack of summands.
	 * @param b  The pack of addends.
	 *
	 * @return   The resultant pack, as a + b.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator + (const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return a.impl + b.impl;
	}

	/**
	 * Adds a pack with a scalar in an entry-wise way.
	 *
	 * @tparam   The scalar type of the packs.
	 * @tparam   The SIMD kind of the packs.
	 * 
	 * @param a  The pack of summands.
	 * @param b  The scalar addend.
	 *
	 * @return   The resultant pack, as a + b.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator + (const simd_pack<T, Kind>& a, const T b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return a.impl + impl_t(b);
	}

	/**
	 * Adds a pack with a scalar in an entry-wise way.
	 *
	 * @tparam   The scalar type of the packs.
	 * @tparam   The SIMD kind of the packs.
	 * 
	 * @param a  The scalar summand.
	 * @param b  The pack of addends.
	 *
	 * @return   The resultant pack, as a + b.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator + (const T a, const simd_pack<T, Kind>& b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return impl_t(a) + b.impl;
	}

	/**
	 * Subtracts two packs in an entry-wise way.
	 *
	 * @tparam   The scalar type of the packs.
	 * @tparam   The SIMD kind of the packs.
	 * 
	 * @param a  The pack of summands.
	 * @param b  The pack of addends.
	 *
	 * @return   The resultant pack, as a - b.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator - (const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return a.impl - b.impl;
	}

	/**
	 * Subtracts a scalar b from a pack a in an entry-wise way.
	 *
	 * @tparam   The scalar type of the packs.
	 * @tparam   The SIMD kind of the packs.
	 * 
	 * @param a  The pack of minuends.
	 * @param b  The scalar subtrahend.
	 *
	 * @return   The resultant pack, as a - b.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator - (const simd_pack<T, Kind>& a, const T b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return a.impl - impl_t(b);
	}

	/**
	 * Subtracts a pack b from a scalar a in an entry-wise way.
	 *
	 * @tparam   The scalar type of the packs.
	 * @tparam   The SIMD kind of the packs.
	 * 
	 * @param a  The scalar minuend.
	 * @param b  The pack of subtrahends.
	 *
	 * @return   The resultant pack, as a - b.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator - (const T a, const simd_pack<T, Kind>& b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return impl_t(a) - b.impl;
	}

	/**
	 * Multiplies two packs in an entry-wise way.
	 *
	 * @tparam   The scalar type of the packs.
	 * @tparam   The SIMD kind of the packs.
	 * 
	 * @param a  The pack of multiplicands.
	 * @param b  The pack of multipliers.
	 *
	 * @return   The resultant pack, as a * b.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator * (const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return a.impl * b.impl;
	}

	/**
	 * Multiplies a pack with a scalar in an entry-wise way.
	 *
	 * @tparam   The scalar type of the packs.
	 * @tparam   The SIMD kind of the packs.
	 * 
	 * @param a  The pack of multiplicands.
	 * @param b  The scalar as a multiplier.
	 *
	 * @return   The resultant pack, as a * b.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator * (const simd_pack<T, Kind>& a, const T b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return a.impl * impl_t(b);
	}

	/**
	 * Multiplies a pack with a scalar in an entry-wise way.
	 *
	 * @tparam   The scalar type of the packs.
	 * @tparam   The SIMD kind of the packs.
	 * 
	 * @param a  The scalar as a multiplicand.
	 * @param b  The pack of multipliers
	 *
	 * @return   The resultant pack, as a * b.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator * (const T a, const simd_pack<T, Kind>& b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return impl_t(a) * b.impl;
	}

	/**
	 * Divides a pack b from another pack a in an entry-wise way.
	 *
	 * @tparam   The scalar type of the packs.
	 * @tparam   The SIMD kind of the packs.
	 * 
	 * @param a  The pack of dividends.
	 * @param b  The pack of divisors.
	 *
	 * @return   The resultant pack, as a / b.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator / (const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return a.impl / b.impl;
	}

	/**
	 * Divides a scalar b from a pack a in an entry-wise way.
	 *
	 * @tparam   The scalar type of the packs.
	 * @tparam   The SIMD kind of the packs.
	 * 
	 * @param a  The pack of dividends.
	 * @param b  The scalar divisor.
	 *
	 * @return   The resultant pack, as a / b.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator / (const simd_pack<T, Kind>& a, const T b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return a.impl / impl_t(b);
	}

	/**
	 * Divides a pack b from a scalar a in an entry-wise way.
	 *
	 * @tparam   The scalar type of the packs.
	 * @tparam   The SIMD kind of the packs.
	 * 
	 * @param a  The scalar dividend.
	 * @param b  The pack of divisors.
	 *
	 * @return   The resultant pack, as a / b.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator / (const T a, const simd_pack<T, Kind>& b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return impl_t(a) / b.impl;
	}

	/**
	 * Negates a pack.
	 *
	 * @tparam   The scalar type of the packs.
	 * @tparam   The SIMD kind of the packs.
	 *
	 * @param a  The input pack.
	 *
	 * @return   The resultant pack, as -a.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator - (const simd_pack<T, Kind>& a)
	{
		return - a.impl;
	}


	/**
	 * Evaluates the absolute values in an entry-wise way.
	 *
	 * @tparam   The scalar type of the packs.
	 * @tparam   The SIMD kind of the packs.
	 *
	 * @param a  The input pack.
	 *
	 * @return   The resultant pack, as |a|.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> abs(const simd_pack<T, Kind>& a)
	{
		return abs(a.impl);
	}

	/**
	 * Selects the smaller values between two packs in an entry-wise way.
	 *
	 * @tparam   The scalar type of the packs.
	 * @tparam   The SIMD kind of the packs.
	 *
	 * @param a   The first input pack.
	 * @param b   The second input pack.
	 *
	 * @return    The resultant pack.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> vmin(const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return vmin(a.impl, b.impl);
	}

	/**
	 * Selects the larger values between two packs in an entry-wise way.
	 *
	 * @tparam   The scalar type of the packs.
	 * @tparam   The SIMD kind of the packs.
	 *
	 * @param a   The first input pack.
	 * @param b   The second input pack.
	 *
	 * @return    The resultant pack.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> vmax(const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return vmax(a.impl, b.impl);
	}

	/**
	 * Calculates the squared roots in an entry-wise way.
	 *
	 * @tparam   The scalar type of the packs.
	 * @tparam   The SIMD kind of the packs.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as sqrt(a), i.e. a^(1/2).
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> sqrt(const simd_pack<T, Kind>& a)
	{
		return sqrt(a.impl);
	}

	/**
	 * Calculates the reciprocals in an entry-wise way.
	 *
	 * @tparam   The scalar type of the packs.
	 * @tparam   The SIMD kind of the packs.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as 1 / a.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> rcp(const simd_pack<T, Kind>& a)
	{
		return rcp(a.impl);
	}

	/**
	 * Calculates the squared roots of the reciprocals in an 
	 * entry-wise way.
	 *
	 * @tparam   The scalar type of the packs.
	 * @tparam   The SIMD kind of the packs.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as 1 / sqrt(a), i.e. a^(-1/2).
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> rsqrt(const simd_pack<T, Kind>& a)
	{
		return rsqrt(a.impl);
	}

	/**
	 * Calculates the squares in an entry-wise way.
	 *
	 * @tparam   The scalar type of the packs.
	 * @tparam   The SIMD kind of the packs.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as a^2.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> sqr(const simd_pack<T, Kind>& a)
	{
		return sqr(a.impl);
	}

	/**
	 * Calculates the cubes in an entry-wise way.
	 *
	 * @tparam   The scalar type of the packs.
	 * @tparam   The SIMD kind of the packs.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as a^2.
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> cube(const simd_pack<T, Kind>& a)
	{
		return cube(a.impl);
	}


	/**
	 * Calculates the floor values in an entry-wise way.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as floor(a).
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> floor(const simd_pack<T, Kind>& a)
	{
		return floor(a.impl);
	}

	/**
	 * Calculates the ceil values in an entry-wise way.
	 *
	 * @param a   The input pack.
	 *
	 * @return    The resultant pack, as ceil(a).
	 */
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> ceil(const simd_pack<T, Kind>& a)
	{
		return ceil(a.impl);
	}

	/** @} */  // arith_generic

}


#ifdef _MSC_VER
#pragma warning(pop)
#endif


#endif /* SIMD_ARITH_H_ */
