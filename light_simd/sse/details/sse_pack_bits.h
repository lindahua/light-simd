/**
 * @file sse_pack_bits.h
 *
 * Internal implementation for SSE packs
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_PACK_BITS_H_
#define LSIMD_SSE_PACK_BITS_H_

#include "../sse_base.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4141)
#endif

namespace lsimd {  namespace sse {


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


	/********************************************
	 *
	 *  duplication
	 *
	 ********************************************/

	LSIMD_ENSURE_INLINE
	inline __m128 f32_dup_low(__m128 v) // [0, 1, 0, 1]
	{
		return _mm_movelh_ps(v, v);
	}

	LSIMD_ENSURE_INLINE
	inline __m128 f32_dup_high(__m128 v) // [2, 3, 2, 3]
	{
		return _mm_movehl_ps(v, v);
	}


	LSIMD_ENSURE_INLINE
	inline __m128 f32_dup2_low(__m128 v) // [0, 0, 2, 2]
	{
#if (defined(LSIMD_HAS_SSE3))
		return _mm_moveldup_ps(v);
#else
		return _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 0, 0));
#endif
	}

	LSIMD_ENSURE_INLINE
	inline __m128 f32_dup2_high(__m128 v) // [1, 1, 3, 3]
	{
#if (defined(LSIMD_HAS_SSE3))
		return _mm_movehdup_ps(v);
#else
		return _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 1, 1));
#endif
	}


	LSIMD_ENSURE_INLINE
	inline __m128d f64_dup_low(__m128d v) // [0, 0]
	{
#if (defined(LSIMD_HAS_SSE3))
		return _mm_movedup_pd(v);
#else
		return _mm_shuffle_pd(v, v, _MM_SHUFFLE2(0, 0));
#endif
	}

	LSIMD_ENSURE_INLINE
	inline __m128d f64_dup_high(__m128d v) // [1, 1]
	{
		return _mm_shuffle_pd(v, v, _MM_SHUFFLE2(1, 1));
	}


	/********************************************
	 *
	 *  sum / max / min
	 *
	 ********************************************/

	// sum

	LSIMD_ENSURE_INLINE
	inline f32 f32_sum(__m128 p)
	{
		__m128 p1 = f32_dup_high(p);
		p1 = _mm_add_ps(p, p1);

		__m128 p2 = f32_dup2_high(p1);
		p1 = _mm_add_ss(p1, p2);

		return _mm_cvtss_f32(p1);
	}

	LSIMD_ENSURE_INLINE
	inline f64 f64_sum(__m128d p)
	{
		__m128d p1 = f64_dup_high(p);
		p1 = _mm_add_sd(p, p1);

		return _mm_cvtsd_f64(p1);
	}


	template<int I> inline f32 f32_partial_sum(__m128 p);
	template<int I> inline f64 f64_partial_sum(__m128d p);

	template<>
	LSIMD_ENSURE_INLINE
	inline f32 f32_partial_sum<1>(__m128 p)
	{
		return _mm_cvtss_f32(p);
	}

	template<>
	LSIMD_ENSURE_INLINE
	inline f32 f32_partial_sum<2>(__m128 p)
	{
		__m128 p1 = f32_dup2_high(p);
		p1 = _mm_add_ss(p1, p);

		return _mm_cvtss_f32(p1);
	}

	template<>
	LSIMD_ENSURE_INLINE
	inline f32 f32_partial_sum<3>(__m128 p)
	{
		__m128 p1 = f32_dup_high(p);
		p1 = _mm_add_ss(p, p1);

		__m128 p2 = f32_dup2_high(p);
		p1 = _mm_add_ss(p1, p2);

		return _mm_cvtss_f32(p1);
	}

	template<>
	LSIMD_ENSURE_INLINE
	inline f64 f64_partial_sum<1>(__m128d p)
	{
		return _mm_cvtsd_f64(p);
	}

	// max

	LSIMD_ENSURE_INLINE
	inline f32 f32_max(__m128 p)
	{
		__m128 p1 = f32_dup_high(p);
		p1 = _mm_max_ps(p, p1);

		__m128 p2 = f32_dup2_high(p1);
		p1 = _mm_max_ss(p1, p2);

		return _mm_cvtss_f32(p1);
	}

	LSIMD_ENSURE_INLINE
	inline f64 f64_max(__m128d p)
	{
		__m128d p1 = f64_dup_high(p);
		p1 = _mm_max_sd(p, p1);

		return _mm_cvtsd_f64(p1);
	}


	template<int I> inline f32 f32_partial_max(__m128 p);
	template<int I> inline f64 f64_partial_max(__m128d p);

	template<>
	LSIMD_ENSURE_INLINE
	inline f32 f32_partial_max<1>(__m128 p)
	{
		return _mm_cvtss_f32(p);
	}

	template<>
	LSIMD_ENSURE_INLINE
	inline f32 f32_partial_max<2>(__m128 p)
	{
		__m128 p1 = f32_dup2_high(p);
		p1 = _mm_max_ss(p1, p);

		return _mm_cvtss_f32(p1);
	}

	template<>
	LSIMD_ENSURE_INLINE
	inline f32 f32_partial_max<3>(__m128 p)
	{
		__m128 p1 = f32_dup_high(p);
		p1 = _mm_max_ss(p, p1);

		__m128 p2 = f32_dup2_high(p);
		p1 = _mm_max_ss(p1, p2);

		return _mm_cvtss_f32(p1);
	}

	template<>
	LSIMD_ENSURE_INLINE
	inline f64 f64_partial_max<1>(__m128d p)
	{
		return _mm_cvtsd_f64(p);
	}


	// min

	LSIMD_ENSURE_INLINE
	inline f32 f32_min(__m128 p)
	{
		__m128 p1 = f32_dup_high(p);
		p1 = _mm_min_ps(p, p1);

		__m128 p2 = f32_dup2_high(p1);
		p1 = _mm_min_ss(p1, p2);

		return _mm_cvtss_f32(p1);
	}

	LSIMD_ENSURE_INLINE
	inline f64 f64_min(__m128d p)
	{
		__m128d p1 = f64_dup_high(p);
		p1 = _mm_min_sd(p, p1);

		return _mm_cvtsd_f64(p1);
	}


	template<int I> inline f32 f32_partial_min(__m128 p);
	template<int I> inline f64 f64_partial_min(__m128d p);

	template<>
	LSIMD_ENSURE_INLINE
	inline f32 f32_partial_min<1>(__m128 p)
	{
		return _mm_cvtss_f32(p);
	}

	template<>
	LSIMD_ENSURE_INLINE
	inline f32 f32_partial_min<2>(__m128 p)
	{
		__m128 p1 = f32_dup2_high(p);
		p1 = _mm_min_ss(p1, p);

		return _mm_cvtss_f32(p1);
	}

	template<>
	LSIMD_ENSURE_INLINE
	inline f32 f32_partial_min<3>(__m128 p)
	{
		__m128 p1 = f32_dup_high(p);
		p1 = _mm_min_ss(p, p1);

		__m128 p2 = f32_dup2_high(p);
		p1 = _mm_min_ss(p1, p2);

		return _mm_cvtss_f32(p1);
	}

	template<>
	LSIMD_ENSURE_INLINE
	inline f64 f64_partial_min<1>(__m128d p)
	{
		return _mm_cvtsd_f64(p);
	}


} }

#ifdef _MSC_VER
#pragma warning(pop)
#endif


#endif 
