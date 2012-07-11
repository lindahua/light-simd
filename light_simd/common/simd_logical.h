/*
 * @file simd_logical.h
 *
 * Comparison and logical operations on Generic SIMD packs.
 *
 * @author Dahua Lin
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

	/********************************************
	 *
	 *  Comparison
	 *
	 ********************************************/

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator == (const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return a.impl == b.impl;
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator != (const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return a.impl != b.impl;
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator < (const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return a.impl < b.impl;
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator <= (const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return a.impl <= b.impl;
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator > (const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return a.impl > b.impl;
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator >= (const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return a.impl >= b.impl;
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator ~ (const simd_pack<T, Kind>& a)
	{
		return ~a.impl;
	}


	/********************************************
	 *
	 *  Comparison
	 *
	 ********************************************/

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator & (const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return a.impl & b.impl;
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator | (const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return a.impl | b.impl;
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator ^ (const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return a.impl ^ b.impl;
	}
	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> cond(const simd_pack<T, Kind>& msk, const simd_pack<T, Kind>& a, const simd_pack<T, Kind>& b)
	{
		return cond(msk.impl, a.impl, b.impl);
	}


}

#endif 
