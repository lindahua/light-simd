/*
 * @file sse_logical.h
 *
 * @brief SSE Comparison and logical operators on SSE Packs.
 *
 * @author Dahua Lin
 */

#ifndef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_LOGICAL_H_
#define LSIMD_SSE_LOGICAL_H_

#include <light_simd/sse/sse_pack.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4141)
#endif

namespace lsimd
{

	/********************************************
	 *
	 *  Comparison operations
	 *
	 ********************************************/

	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator == (const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_cmpeq_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator == (const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_cmpeq_pd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator != (const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_cmpneq_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator != (const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_cmpneq_pd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator < (const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_cmplt_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator < (const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_cmplt_pd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator <= (const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_cmple_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator <= (const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_cmple_pd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator > (const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_cmpgt_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator > (const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_cmpgt_pd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator >= (const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_cmpge_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator >= (const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_cmpge_pd(a.v, b.v);
	}


	/********************************************
	 *
	 *  Bitwise logical operations
	 *
	 ********************************************/

	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator ~ (const sse_f32pk& a)
	{
		return _mm_xor_ps(a.v, _mm_castsi128_ps(sse_internal::all_one_bits()));
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator ~ (const sse_f64pk& a)
	{
		return _mm_xor_pd(a.v, _mm_castsi128_pd(sse_internal::all_one_bits()));
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator & (const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_and_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator & (const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_and_pd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator | (const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_or_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator | (const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_or_pd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator ^ (const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_xor_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator ^ (const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_xor_pd(a.v, b.v);
	}



	/********************************************
	 *
	 *  Conditional selection
	 *
	 ********************************************/

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


	LSIMD_ENSURE_INLINE
	inline sse_f32pk cond(const sse_f32pk& msk, const sse_f32pk& a, const sse_f32pk& b)
	{
#ifdef LSIMD_HAS_SSE4_1
		return cond_sse4(msk, a, b);
#else
		return cond_sse2(msk, a, b);
#endif
	}


	LSIMD_ENSURE_INLINE
	inline sse_f64pk cond(const sse_f64pk& msk, const sse_f64pk& a, const sse_f64pk& b)
	{
#ifdef LSIMD_HAS_SSE4_1
		return cond_sse4(msk, a, b);
#else
		return cond_sse2(msk, a, b);
#endif
	}

}


#endif 





