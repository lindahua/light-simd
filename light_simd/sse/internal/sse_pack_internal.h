/**
 * @file sse_pack_internal.h
 *
 * Internal implementation for SSE packs
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_PACK_INTERNAL_H_
#define LSIMD_SSE_PACK_INTERNAL_H_

#include <light_simd/sse/sse_base.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4141)
#endif

namespace lsimd {  namespace sse_internal {


	/********************************************
	 *
	 *  partial load / store
	 *
	 ********************************************/

	LSIMD_ENSURE_INLINE
	inline __m128 partial_load(const f32 *x, int_<0>)
	{
		return _mm_setzero_ps();
	}

	LSIMD_ENSURE_INLINE
	inline void partial_store(f32 *x, __m128 p, int_<0>)
	{
	}

	LSIMD_ENSURE_INLINE
	inline __m128 partial_load(const f32 *x, int_<1>)
	{
		return _mm_load_ss(x);
	}

	LSIMD_ENSURE_INLINE
	inline void partial_store(f32 *x, __m128 p, int_<1>)
	{
		_mm_store_ss(x, p);
	}

	LSIMD_ENSURE_INLINE
	inline __m128 partial_load(const f32 *x, int_<2>)
	{
		return _mm_castsi128_ps(
				_mm_loadl_epi64(reinterpret_cast<const __m128i*>(x)));
	}

	LSIMD_ENSURE_INLINE
	inline void partial_store(f32 *x, __m128 p, int_<2>)
	{
		_mm_storel_epi64(reinterpret_cast<__m128i*>(x), _mm_castps_si128(p));
	}

	LSIMD_ENSURE_INLINE
	inline __m128 partial_load(const f32 *x, int_<3>)
	{
		__m128 a01 = _mm_castsi128_ps(
				_mm_loadl_epi64(reinterpret_cast<const __m128i*>(x)));
		__m128 a2 = _mm_load_ss(x+2);

		return _mm_movelh_ps(a01, a2);
	}

	LSIMD_ENSURE_INLINE
	inline void partial_store(f32 *x, __m128 p, int_<3>)
	{
		_mm_storel_epi64(
				reinterpret_cast<__m128i*>(x), _mm_castps_si128(p));
				_mm_store_ss(x+2, _mm_movehl_ps(p, p));
	}

	LSIMD_ENSURE_INLINE
	inline __m128 partial_load(const f32 *x, int_<4>)
	{
		return _mm_loadu_ps(x);
	}

	LSIMD_ENSURE_INLINE
	inline void partial_store(f32 *x, __m128 p, int_<4>)
	{
		_mm_storeu_ps(x, p);
	}


	LSIMD_ENSURE_INLINE
	inline __m128d partial_load(const f64 *x, int_<0>)
	{
		return _mm_setzero_pd();
	}

	LSIMD_ENSURE_INLINE
	inline void partial_store(f64 *x, __m128d p, int_<0>)
	{
	}


	LSIMD_ENSURE_INLINE
	inline __m128d partial_load(const f64 *x, int_<1>)
	{
		return _mm_load_sd(x);
	}

	LSIMD_ENSURE_INLINE
	inline void partial_store(f64 *x, __m128d p, int_<1>)
	{
		_mm_store_sd(x, p);
	}


	LSIMD_ENSURE_INLINE
	inline __m128d partial_load(const f64 *x, int_<2>)
	{
		return _mm_loadu_pd(x);
	}

	LSIMD_ENSURE_INLINE
	inline void partial_store(f64 *x, __m128d p, int_<2>)
	{
		_mm_storeu_pd(x, p);
	}


	/********************************************
	 *
	 *  Entry extraction
	 *
	 ********************************************/


#if defined(LSIMD_HAS_SSE4_1)

	LSIMD_ENSURE_INLINE
	inline f32 f32p_extract(__m128 a, int_<0>)
	{
		f32 r;
		_MM_EXTRACT_FLOAT(r, a, 0);
		return r;
	}

	LSIMD_ENSURE_INLINE
	inline f32 f32p_extract(__m128 a, int_<1>)
	{
		f32 r;
		_MM_EXTRACT_FLOAT(r, a, 1);
		return r;
	}

	LSIMD_ENSURE_INLINE
	inline f32 f32p_extract(__m128 a, int_<2>)
	{
		f32 r;
		_MM_EXTRACT_FLOAT(r, a, 2);
		return r;
	}

	LSIMD_ENSURE_INLINE
	inline f32 f32p_extract(__m128 a, int_<3>)
	{
		f32 r;
		_MM_EXTRACT_FLOAT(r, a, 3);
		return r;
	}

#else

	LSIMD_ENSURE_INLINE
	inline f32 f32p_extract(__m128 a, int_<0>)
	{
		return _mm_cvtss_f32(a);
	}

	LSIMD_ENSURE_INLINE
	inline f32 f32p_extract(__m128 a, int_<1>)
	{
		return _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(a), 4)));
	}

	LSIMD_ENSURE_INLINE
	inline f32 f32p_extract<2>(__m128 a, int_<2>)
	{
		return _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(a), 8)));
	}

	LSIMD_ENSURE_INLINE
	inline f32 f32p_extract<3>(__m128 a, int_<3>)
	{
		return _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(a), 12)));
	}


#endif

	LSIMD_ENSURE_INLINE
	inline f64 f64p_extract(__m128d a, int_<0>)
	{
		return _mm_cvtsd_f64(a);
	}

	LSIMD_ENSURE_INLINE
	inline f64 f64p_extract(__m128d a, int_<1>)
	{
		return _mm_cvtsd_f64(_mm_castsi128_pd(_mm_srli_si128(_mm_castpd_si128(a), 8)));
	}

} }

#ifdef _MSC_VER
#pragma warning(pop)
#endif


#endif 
