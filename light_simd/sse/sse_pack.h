/**
 * @file simd_pack.h
 *
 * @brief The SSE pack classes and a set of convenient routines.
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_PACK_H_
#define LSIMD_SSE_PACK_H_

#include "internal/sse_pack_internal.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4141)
#endif

namespace lsimd
{
	/******************************************************
	 *
	 *  traits
	 *
	 ******************************************************/

	template<>
	struct simd_traits<float, sse_kind>
	{
		typedef float scalar_type;
		static const unsigned int pack_width = 4;
	};


	template<>
	struct simd_traits<double, sse_kind>
	{
		typedef double scalar_type;
		static const unsigned int pack_width = 2;
	};


	/******************************************************
	 *
	 *  SSE float
	 *
	 ******************************************************/


	/**
	 * @brief SSE pack with four single-precision real values.
	 */
	template<>
	struct simd_pack<f32, sse_kind>
	{
		typedef f32 value_type;
		typedef __m128 intern_type;

		static const unsigned int pack_width = 4;

		union
		{
			__m128 v;
			LSIMD_ALIGN_SSE f32 e[4];
		};


		// constructors

		LSIMD_ENSURE_INLINE simd_pack() { }

		LSIMD_ENSURE_INLINE simd_pack(const __m128& v_) : v(v_) { }

		LSIMD_ENSURE_INLINE simd_pack( tag::all_zeros )
		{
			v = _mm_setzero_ps();
		}

		LSIMD_ENSURE_INLINE simd_pack( tag::all_nonzeros )
		{
			v = _mm_castsi128_ps(_mm_set1_epi32((int)0xffffffff));
		}

		LSIMD_ENSURE_INLINE explicit simd_pack(const f32& x)
		{
			v = _mm_set1_ps(x);
		}

		LSIMD_ENSURE_INLINE simd_pack(const f32& e0, const f32& e1, const f32& e2, const f32& e3)
		{
			v = _mm_set_ps(e3, e2, e1, e0);
		}

		LSIMD_ENSURE_INLINE simd_pack(bool b0, bool b1, bool b2, bool b3)
		{
			const int i0 = b0 ? (int)(0xffffffff) : 0;
			const int i1 = b1 ? (int)(0xffffffff) : 0;
			const int i2 = b2 ? (int)(0xffffffff) : 0;
			const int i3 = b3 ? (int)(0xffffffff) : 0;

			v = _mm_castsi128_ps(_mm_set_epi32(i3, i2, i1, i0));
		}

		LSIMD_ENSURE_INLINE simd_pack(const f32* a, tag::aligned)
		{
			v = _mm_load_ps(a);
		}

		LSIMD_ENSURE_INLINE simd_pack(const f32* a, tag::unaligned)
		{
			v = _mm_loadu_ps(a);
		}


		// getters

		LSIMD_ENSURE_INLINE unsigned int width() const
		{
			return pack_width;
		}

		LSIMD_ENSURE_INLINE __m128 intern() const
		{
			return v;
		}

		//  set, load, and store

		LSIMD_ENSURE_INLINE void set_zeros()
		{
			v = _mm_setzero_ps();
		}

		LSIMD_ENSURE_INLINE void set(const f32& x)
		{
			v = _mm_set1_ps(x);
		}

		LSIMD_ENSURE_INLINE void set(const f32& e0, const f32& e1, const f32& e2, const f32& e3)
		{
			v = _mm_set_ps(e3, e2, e1, e0);
		}

		LSIMD_ENSURE_INLINE void load(const f32* a, tag::aligned)
		{
			v = _mm_load_ps(a);
		}

		LSIMD_ENSURE_INLINE void load(const f32* a, tag::unaligned)
		{
			v = _mm_loadu_ps(a);
		}

		LSIMD_ENSURE_INLINE void store(f32* a, tag::aligned) const
		{
			_mm_store_ps(a, v);
		}

		LSIMD_ENSURE_INLINE void store(f32* a, tag::unaligned) const
		{
			_mm_storeu_ps(a, v);
		}

		template<int I>
		LSIMD_ENSURE_INLINE void partial_load(const f32 *a, int_<I>)
		{
			v = sse_internal::partial_load(a, int_<I>());
		}

		template<int I>
		LSIMD_ENSURE_INLINE void partial_store(f32 *a, int_<I>) const
		{
			sse_internal::partial_store(a, v, int_<I>());
		}


		// entry manipulation

		LSIMD_ENSURE_INLINE f32 to_scalar() const
		{
			return _mm_cvtss_f32(v);
		}

		template<int I>
		LSIMD_ENSURE_INLINE f32 extract() const
		{
			return sse_internal::f32p_extract(v, int_<I>());
		}

		// Only for debug

		LSIMD_ENSURE_INLINE bool test_equal(f32 e0, f32 e1, f32 e2, f32 e3) const
		{
			return e[0] == e0 && e[1] == e1 && e[2] == e2 && e[3] == e3;
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return test_equal(r[0], r[1], r[2], r[3]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("(");
			std::printf(fmt, e[0]); std::printf(", ");
			std::printf(fmt, e[1]); std::printf(", ");
			std::printf(fmt, e[2]); std::printf(", ");
			std::printf(fmt, e[3]);
			std::printf(")");
		}

	};


	/******************************************************
	 *
	 *  SSE double
	 *
	 ******************************************************/

	/**
	 * @brief SSE pack with two double-precision real values.
	 */
	template<>
	struct simd_pack<f64, sse_kind>
	{
		typedef f64 value_type;
		typedef __m128d intern_type;

		static const unsigned int pack_width = 2;

		union
		{
			__m128d v;
			LSIMD_ALIGN_SSE f64 e[2];
		};

		
		// constructors

		LSIMD_ENSURE_INLINE simd_pack() { }

		LSIMD_ENSURE_INLINE simd_pack(const __m128d& v_) : v(v_) { }

		LSIMD_ENSURE_INLINE simd_pack( tag::all_zeros )
		{
			v = _mm_setzero_pd();
		}

		LSIMD_ENSURE_INLINE simd_pack( tag::all_nonzeros )
		{
			v = _mm_castsi128_pd(_mm_set1_epi32((int)0xffffffff));
		}

		LSIMD_ENSURE_INLINE explicit simd_pack(const f64& x)
		{
			v = _mm_set1_pd(x);
		}

		LSIMD_ENSURE_INLINE simd_pack(const f64& e0, const f64& e1)
		{
			v = _mm_set_pd(e1, e0);
		}

		LSIMD_ENSURE_INLINE simd_pack(bool b0, bool b1)
		{
			int i0, i1, i2, i3;
			i0 = i1 = (b0 ? (int)(0xffffffff) : 0);
			i2 = i3 = (b1 ? (int)(0xffffffff) : 0);
			v = _mm_castsi128_pd(_mm_set_epi32(i3, i2, i1, i0));
		}

		LSIMD_ENSURE_INLINE simd_pack(const f64* a, tag::aligned)
		{
			v = _mm_load_pd(a);
		}

		LSIMD_ENSURE_INLINE simd_pack(const f64* a, tag::unaligned)
		{
			v = _mm_loadu_pd(a);
		}


		// getters

		LSIMD_ENSURE_INLINE unsigned int width() const
		{
			return pack_width;
		}

		LSIMD_ENSURE_INLINE intern_type intern() const
		{
			return v;
		}

		// set, load, and store

		LSIMD_ENSURE_INLINE void set_zero()
		{
			v = _mm_setzero_pd();
		}

		LSIMD_ENSURE_INLINE void set(const f64& x)
		{
			v = _mm_set1_pd(x);
		}

		LSIMD_ENSURE_INLINE void set(const f64& e0, const f64& e1)
		{
			v = _mm_set_pd(e1, e0);
		}

		LSIMD_ENSURE_INLINE void load(const f64* a, tag::aligned)
		{
			v = _mm_load_pd(a);
		}

		LSIMD_ENSURE_INLINE void load(const f64* a, tag::unaligned)
		{
			v = _mm_loadu_pd(a);
		}

		LSIMD_ENSURE_INLINE void store(f64* a, tag::aligned) const
		{
			_mm_store_pd(a, v);
		}

		LSIMD_ENSURE_INLINE void store(f64* a, tag::unaligned) const
		{
			_mm_storeu_pd(a, v);
		}

		template<int I>
		LSIMD_ENSURE_INLINE void partial_load(const f64 *a, int_<I>)
		{
			v = sse_internal::partial_load(a, int_<I>());
		}

		template<int I>
		LSIMD_ENSURE_INLINE void partial_store(f64 *a, int_<I>) const
		{
			sse_internal::partial_store(a, v, int_<I>());
		}


		// entry manipulation

		LSIMD_ENSURE_INLINE f64 to_scalar() const
		{
			return _mm_cvtsd_f64(v);
		}

		template<int I>
		LSIMD_ENSURE_INLINE f64 extract() const
		{
			return sse_internal::f64p_extract(v, int_<I>());
		}

		// Only for debug

		LSIMD_ENSURE_INLINE bool test_equal(f64 e0, f64 e1) const
		{
			return e[0] == e0 && e[1] == e1;
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return test_equal(r[0], r[1]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("(");
			std::printf(fmt, e[0]); std::printf(", ");
			std::printf(fmt, e[1]);
			std::printf(")");
		}

	}; // end struct simd_pack<f64>


	// typedefs

	typedef simd_pack<f32, sse_kind> sse_f32pk;
	typedef simd_pack<f64, sse_kind> sse_f64pk;


	// Shuffling

	template<int I0, int I1, int I2, int I3>
	LSIMD_ENSURE_INLINE
	inline sse_f32pk shuffle(const sse_f32pk& a, const sse_f32pk& b, pat4_<I0, I1, I2, I3>)
	{
		return _mm_shuffle_ps(a.v, b.v, _MM_SHUFFLE(I3, I2, I1, I0));
	}

	template<int I0, int I1>
	LSIMD_ENSURE_INLINE
	inline sse_f64pk shuffle(const sse_f64pk& a, const sse_f64pk& b, pat2_<I0, I1>)
	{
		return _mm_shuffle_pd(a.v, b.v, _MM_SHUFFLE2(I1, I0));
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk shuffle(const sse_f32pk& a, const sse_f32pk& b, pat4_<0, 1, 0, 1>)
	{
		return _mm_movelh_ps(a.v, b.v);
	}

	template<int I0, int I1>
	LSIMD_ENSURE_INLINE
	inline sse_f32pk shuffle(const sse_f32pk& a, const sse_f32pk& b, pat4_<2, 3, 2, 3>)
	{
		return _mm_movehl_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk shuffle(const sse_f64pk& a, const sse_f64pk& b, pat2_<0, 0>)
	{
		return _mm_unpacklo_pd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk shuffle(const sse_f64pk& a, const sse_f64pk& b, pat2_<1, 1>)
	{
		return _mm_unpackhi_pd(a.v, b.v);
	}

	template<int I0, int I1, int I2, int I3>
	LSIMD_ENSURE_INLINE
	inline sse_f32pk swizzle(const sse_f32pk& a, pat4_<I0, I1, I2, I3>)
	{
		return shuffle(a, a, pat4_<I0, I1, I2, I3>());
	}

	template<int I0, int I1>
	LSIMD_ENSURE_INLINE
	inline sse_f64pk swizzle(const sse_f64pk& a, pat2_<I0, I1>)
	{
		return shuffle(a, a, pat2_<I0, I1>());
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk swizzle(const sse_f32pk& a, pat4_<0, 0, 1, 1>)
	{
		return _mm_unpacklo_ps(a.v, a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk swizzle(const sse_f32pk& a, pat4_<2, 2, 3, 3>)
	{
		return _mm_unpackhi_ps(a.v, a.v);
	}

#ifdef LSIMD_HAS_SSE3

	LSIMD_ENSURE_INLINE
	inline sse_f32pk swizzle(const sse_f32pk& a, pat4_<0, 0, 2, 2>)
	{
		return _mm_moveldup_ps(a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk swizzle(const sse_f32pk& a, pat4_<1, 1, 3, 3>)
	{
		return _mm_movehdup_ps(a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk swizzle(const sse_f64pk& a, pat2_<0, 0>)
	{
		return _mm_movedup_pd(a.v);
	}

#endif


	template<int I>
	LSIMD_ENSURE_INLINE
	inline sse_f32pk broadcast(const sse_f32pk& a, int_<I>)
	{
		return swizzle(a, pat4_<I, I, I, I>());
	}

	template<int I>
	LSIMD_ENSURE_INLINE
	inline sse_f64pk broadcast(const sse_f64pk& a, int_<I>)
	{
		return swizzle(a, pat2_<I, I>());
	}

}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif /* SSE_PACK_H_ */

