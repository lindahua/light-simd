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

	template<int I> inline __m128  partial_load(const f32 *x);
	template<int I> inline __m128d partial_load(const f64 *x);
	template<int I> inline void partial_store(f32 *x, __m128  p);
	template<int I> inline void partial_store(f64 *x, __m128d p);


	template<>
	LSIMD_ENSURE_INLINE
	inline __m128 partial_load<1>(const f32 *x)
	{
		return _mm_load_ss(x);
	}

	template<>
	LSIMD_ENSURE_INLINE
	inline __m128 partial_load<2>(const f32 *x)
	{
		return _mm_castsi128_ps(
				_mm_loadl_epi64(reinterpret_cast<const __m128i*>(x)));
	}

	template<>
	LSIMD_ENSURE_INLINE
	inline __m128 partial_load<3>(const f32 *x)
	{
		__m128 a01 = _mm_castsi128_ps(
				_mm_loadl_epi64(reinterpret_cast<const __m128i*>(x)));
		__m128 a2 = _mm_load_ss(x+2);

		return _mm_movelh_ps(a01, a2);
	}

	template<>
	LSIMD_ENSURE_INLINE
	inline __m128d partial_load<1>(const f64 *x)
	{
		return _mm_load_sd(x);
	}


	template<>
	LSIMD_ENSURE_INLINE
	inline void partial_store<1>(f32 *x, __m128 p)
	{
		_mm_store_ss(x, p);
	}

	template<>
	LSIMD_ENSURE_INLINE
	inline void partial_store<2>(f32 *x, __m128 p)
	{
		_mm_storel_epi64(reinterpret_cast<__m128i*>(x), _mm_castps_si128(p));
	}

	template<>
	LSIMD_ENSURE_INLINE
	inline void partial_store<3>(f32 *x, __m128 p)
	{
		_mm_storel_epi64(reinterpret_cast<__m128i*>(x), _mm_castps_si128(p));
		_mm_store_ss(x+2, _mm_movehl_ps(p, p));
	}

	template<>
	LSIMD_ENSURE_INLINE
	inline void partial_store<1>(f64 *x, __m128d p)
	{
		_mm_store_sd(x, p);
	}



	/********************************************
	 *
	 *  Entry extraction
	 *
	 ********************************************/

	template<int I> inline f32 f32p_extract(__m128 a);

	template<int I> inline f64 f64p_extract(__m128d a);


#if defined(LSIMD_HAS_SSE4_1)

	template<>
	LSIMD_ENSURE_INLINE
	inline f32 f32p_extract<0>(__m128 a)
	{
		f32 r;
		_MM_EXTRACT_FLOAT(r, a, 0);
		return r;
	}

	template<>
	LSIMD_ENSURE_INLINE
	inline f32 f32p_extract<1>(__m128 a)
	{
		f32 r;
		_MM_EXTRACT_FLOAT(r, a, 1);
		return r;
	}

	template<>
	LSIMD_ENSURE_INLINE
	inline f32 f32p_extract<2>(__m128 a)
	{
		f32 r;
		_MM_EXTRACT_FLOAT(r, a, 2);
		return r;
	}

	template<>
	LSIMD_ENSURE_INLINE
	inline f32 f32p_extract<3>(__m128 a)
	{
		f32 r;
		_MM_EXTRACT_FLOAT(r, a, 3);
		return r;
	}

#else

	template<>
	LSIMD_ENSURE_INLINE
	inline f32 f32p_extract<0>(__m128 a)
	{
		return _mm_cvtss_f32(a);
	}

	template<>
	LSIMD_ENSURE_INLINE
	inline f32 f32p_extract<1>(__m128 a)
	{
		return _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(a), 4)));
	}

	template<>
	LSIMD_ENSURE_INLINE
	inline f32 f32p_extract<2>(__m128 a)
	{
		return _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(a), 8)));
	}

	template<>
	LSIMD_ENSURE_INLINE
	inline f32 f32p_extract<3>(__m128 a)
	{
		return _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(a), 12)));
	}


#endif

	template<>
	LSIMD_ENSURE_INLINE
	inline f64 f64p_extract<0>(__m128d a)
	{
		return _mm_cvtsd_f64(a);
	}

	template<>
	LSIMD_ENSURE_INLINE
	inline f64 f64p_extract<1>(__m128d a)
	{
		return _mm_cvtsd_f64(_mm_castsi128_pd(_mm_srli_si128(_mm_castpd_si128(a), 8)));
	}

} }

#ifdef _MSC_VER
#pragma warning(pop)
#endif


#endif 
