/**
 * @file avx_pack.h
 *
 * @brief The AVX SIMD pack classes and functions to manipulate them
 *
 * @author Dahua Lin
 */

#ifndef LSIMD_AVX_PACK_H_
#define LSIMD_AVX_PACK_H_

#include "internal/avx_pack_internal.h"

namespace lsimd
{
	/******************************************************
	 *
	 *  traits
	 *
	 ******************************************************/

	template<>
	struct simd_traits<f32, avx_kind>
	{
		typedef f32 scalar_type;
		static const unsigned int pack_width = 8;
	};


	template<>
	struct simd_traits<f64, avx_kind>
	{
		typedef f64 scalar_type;
		static const unsigned int pack_width = 4;
	};


	/******************************************************
	 *
	 *  AVX float
	 *
	 ******************************************************/

	template<>
	struct simd_pack<f32, avx_kind>
	{
		typedef f32 value_type;
		typedef __m256 intern_type;

		static const unsigned int pack_width = 8;

		union
		{
			__m256 v;
			LSIMD_ALIGN_AVX f32 e[8];
		};


		// constructors

		LSIMD_ENSURE_INLINE simd_pack() { }

		LSIMD_ENSURE_INLINE simd_pack(const __m256& v_) : v(v_) { }

		LSIMD_ENSURE_INLINE simd_pack( tag::all_zeros )
		{
			v = _mm256_setzero_ps();
		}

		LSIMD_ENSURE_INLINE simd_pack( tag::all_nonzeros )
		{
			v = _mm256_castsi256_ps(_mm256_set1_epi32((int)0xffffffff));
		}

		LSIMD_ENSURE_INLINE explicit simd_pack(const f32& x)
		{
			v = _mm256_set1_ps(x);
		}

		LSIMD_ENSURE_INLINE simd_pack(
				const f32& e0, const f32& e1, const f32& e2, const f32& e3,
				const f32& e4, const f32& e5, const f32& e6, const f32& e7)
		{
			v = _mm256_set_ps(e7, e6, e5, e4, e3, e2, e1, e0);
		}

		LSIMD_ENSURE_INLINE simd_pack(
				bool b0, bool b1, bool b2, bool b3,
				bool b4, bool b5, bool b6, bool b7)
		{
			const int i0 = b0 ? (int)(0xffffffff) : 0;
			const int i1 = b1 ? (int)(0xffffffff) : 0;
			const int i2 = b2 ? (int)(0xffffffff) : 0;
			const int i3 = b3 ? (int)(0xffffffff) : 0;
			const int i4 = b4 ? (int)(0xffffffff) : 0;
			const int i5 = b5 ? (int)(0xffffffff) : 0;
			const int i6 = b6 ? (int)(0xffffffff) : 0;
			const int i7 = b7 ? (int)(0xffffffff) : 0;

			v = _mm256_castsi256_ps(_mm256_set_epi32(i7, i6, i5, i4, i3, i2, i1, i0));
		}

		LSIMD_ENSURE_INLINE simd_pack(const f32* a, tag::aligned)
		{
			v = _mm256_load_ps(a);
		}

		LSIMD_ENSURE_INLINE simd_pack(const f32* a, tag::unaligned)
		{
			v = _mm256_loadu_ps(a);
		}


		// getters

		LSIMD_ENSURE_INLINE unsigned int width() const
		{
			return pack_width;
		}

		LSIMD_ENSURE_INLINE __m256 intern() const
		{
			return v;
		}

		//  set, load, and store

		LSIMD_ENSURE_INLINE void set_zeros()
		{
			v = _mm256_setzero_ps();
		}

		LSIMD_ENSURE_INLINE void set(const f32& x)
		{
			v = _mm256_set1_ps(x);
		}

		LSIMD_ENSURE_INLINE void set(
				const f32& e0, const f32& e1, const f32& e2, const f32& e3,
				const f32& e4, const f32& e5, const f32& e6, const f32& e7)
		{
			v = _mm256_set_ps(e7, e6, e5, e4, e3, e2, e1, e0);
		}

		LSIMD_ENSURE_INLINE void load(const f32* a, tag::aligned)
		{
			v = _mm256_load_ps(a);
		}

		LSIMD_ENSURE_INLINE void load(const f32* a, tag::unaligned)
		{
			v = _mm256_loadu_ps(a);
		}

		LSIMD_ENSURE_INLINE void store(f32* a, tag::aligned) const
		{
			_mm256_store_ps(a, v);
		}

		LSIMD_ENSURE_INLINE void store(f32* a, tag::unaligned) const
		{
			_mm256_storeu_ps(a, v);
		}

		template<int I>
		LSIMD_ENSURE_INLINE void partial_load(const f32 *a, int_<I>)
		{
			_mm256_maskload_ps(a,
					_mm256_castsi256_ps(avx_internal::partial_mask_i32(int_<I>())));
		}

		template<int I>
		LSIMD_ENSURE_INLINE void partial_store(f32 *a, int_<I>) const
		{
			_mm256_maskstore_ps(a,
					_mm256_castsi256_ps(avx_internal::partial_mask_i32(int_<I>())), v);
		}


		// entry manipulation

		LSIMD_ENSURE_INLINE f32 to_scalar() const
		{
			return _mm_cvtss_f32(_mm256_extractf128_ps(v, 0));
		}

		template<int I>
		LSIMD_ENSURE_INLINE f32 extract() const
		{
			return e[I];
		}

		// Only for debug

		LSIMD_ENSURE_INLINE bool test_equal(
				f32 e0, f32 e1, f32 e2, f32 e3, f32 e4, f32 e5, f32 e6, f32 e7) const
		{
			return  e[0] == e0 && e[1] == e1 && e[2] == e2 && e[3] == e3 &&
					e[4] == e4 && e[5] == e5 && e[6] == e6 && e[7] == e7;
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return test_equal(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("(");
			std::printf(fmt, e[0]); std::printf(", ");
			std::printf(fmt, e[1]); std::printf(", ");
			std::printf(fmt, e[2]); std::printf(", ");
			std::printf(fmt, e[3]); std::printf(", ");
			std::printf(fmt, e[4]); std::printf(", ");
			std::printf(fmt, e[5]); std::printf(", ");
			std::printf(fmt, e[6]); std::printf(", ");
			std::printf(fmt, e[7]);
			std::printf(")");
		}

	};


	/******************************************************
	 *
	 *  AVX double
	 *
	 ******************************************************/

	template<>
	struct simd_pack<f64, avx_kind>
	{
		typedef f64 value_type;
		typedef __m256d intern_type;

		static const unsigned int pack_width = 4;

		union
		{
			__m256d v;
			LSIMD_ALIGN_AVX f64 e[4];
		};


		// constructors

		LSIMD_ENSURE_INLINE simd_pack() { }

		LSIMD_ENSURE_INLINE simd_pack(const __m256d& v_) : v(v_) { }

		LSIMD_ENSURE_INLINE simd_pack( tag::all_zeros )
		{
			v = _mm256_setzero_pd();
		}

		LSIMD_ENSURE_INLINE simd_pack( tag::all_nonzeros )
		{
			v = _mm256_castsi256_pd(_mm256_set1_epi32((int)0xffffffff));
		}

		LSIMD_ENSURE_INLINE explicit simd_pack(const f64& x)
		{
			v = _mm256_set1_pd(x);
		}

		LSIMD_ENSURE_INLINE simd_pack(const f64& e0, const f64& e1, const f64& e2, const f64& e3)
		{
			v = _mm256_set_pd(e3, e2, e1, e0);
		}

		LSIMD_ENSURE_INLINE simd_pack(bool b0, bool b1, bool b2, bool b3)
		{
			int i0, i1, i2, i3, i4, i5, i6, i7;

			i0 = i1 = b0 ? (int)(0xffffffff) : 0;
			i2 = i3 = b1 ? (int)(0xffffffff) : 0;
			i4 = i5 = b2 ? (int)(0xffffffff) : 0;
			i6 = i7 = b3 ? (int)(0xffffffff) : 0;

			v = _mm256_castsi256_ps(_mm256_set_epi32(i7, i6, i5, i4, i3, i2, i1, i0));
		}

		LSIMD_ENSURE_INLINE simd_pack(const f64* a, tag::aligned)
		{
			v = _mm256_load_pd(a);
		}

		LSIMD_ENSURE_INLINE simd_pack(const f64* a, tag::unaligned)
		{
			v = _mm256_loadu_pd(a);
		}


		// getters

		LSIMD_ENSURE_INLINE unsigned int width() const
		{
			return pack_width;
		}

		LSIMD_ENSURE_INLINE __m256d intern() const
		{
			return v;
		}

		//  set, load, and store

		LSIMD_ENSURE_INLINE void set_zeros()
		{
			v = _mm256_setzero_pd();
		}

		LSIMD_ENSURE_INLINE void set(const f64& x)
		{
			v = _mm256_set1_pd(x);
		}

		LSIMD_ENSURE_INLINE void set(const f64& e0, const f64& e1, const f64& e2, const f64& e3)
		{
			v = _mm256_set_pd(e3, e2, e1, e0);
		}

		LSIMD_ENSURE_INLINE void load(const f64* a, tag::aligned)
		{
			v = _mm256_load_pd(a);
		}

		LSIMD_ENSURE_INLINE void load(const f64* a, tag::unaligned)
		{
			v = _mm256_loadu_pd(a);
		}

		LSIMD_ENSURE_INLINE void store(f64* a, tag::aligned) const
		{
			_mm256_store_pd(a, v);
		}

		LSIMD_ENSURE_INLINE void store(f64* a, tag::unaligned) const
		{
			_mm256_storeu_pd(a, v);
		}

		template<int I>
		LSIMD_ENSURE_INLINE void partial_load(const f64 *a, int_<I>)
		{
			_mm256_maskload_pd(a,
					_mm256_castsi256_ps(avx_internal::partial_mask_i32(int_<2 * I>())));
		}

		template<int I>
		LSIMD_ENSURE_INLINE void partial_store(f64 *a, int_<I>) const
		{
			_mm256_maskstore_pd(a,
					_mm256_castsi256_ps(avx_internal::partial_mask_i32(int_<2 * I>())), v);
		}


		// entry manipulation

		LSIMD_ENSURE_INLINE f64 to_scalar() const
		{
			return _mm_cvtsd_f64(_mm256_extractf128_pd(v, 0));
		}

		template<int I>
		LSIMD_ENSURE_INLINE f64 extract() const
		{
			return e[I];
		}

		// Only for debug

		LSIMD_ENSURE_INLINE bool test_equal(f64 e0, f64 e1, f64 e2, f64 e3) const
		{
			return  e[0] == e0 && e[1] == e1 && e[2] == e2 && e[3] == e3;
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
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


	/********************************************
	 *
	 *  Typedef
	 *
	 ********************************************/

	typedef simd_pack<f32, avx_kind> avx_f32pk;
	typedef simd_pack<f64, avx_kind> avx_f64pk;


	/********************************************
	 *
	 *  Shuffling and Swizzling
	 *
	 ********************************************/

	template<int I0, int I1, int I2, int I3>
	LSIMD_ENSURE_INLINE
	inline avx_f32pk shuffle(const avx_f32pk& a, const avx_f32pk& b, pat4_<I0, I1, I2, I3>)
	{
		return _mm256_shuffle_ps(a.v, b.v, _MM_SHUFFLE(I3, I2, I1, I0));
	}

	template<int I0, int I1>
	LSIMD_ENSURE_INLINE
	inline avx_f64pk shuffle(const avx_f64pk& a, const avx_f64pk& b, pat2_<I0, I1>)
	{
		return _mm256_shuffle_pd(a.v, b.v, _MM_SHUFFLE2(I1, I0));
	}

	template<int I0, int I1, int I2, int I3>
	LSIMD_ENSURE_INLINE
	inline avx_f32pk swizzle(const avx_f32pk& a, pat4_<I0, I1, I2, I3>)
	{
		return shuffle(a, a, pat4_<I0, I1, I2, I3>());
	}

	template<int I0, int I1>
	LSIMD_ENSURE_INLINE
	inline avx_f64pk swizzle(const avx_f64pk& a, pat2_<I0, I1>)
	{
		return shuffle(a, a, pat2_<I0, I1>());
	}

	LSIMD_ENSURE_INLINE
	inline avx_f32pk swizzle(const avx_f32pk& a, pat4_<0, 0, 1, 1>)
	{
		return _mm256_unpacklo_ps(a.v, a.v);
	}

	LSIMD_ENSURE_INLINE
	inline avx_f32pk swizzle(const avx_f32pk& a, pat4_<2, 2, 3, 3>)
	{
		return _mm256_unpackhi_ps(a.v, a.v);
	}

	LSIMD_ENSURE_INLINE
	inline avx_f64pk swizzle(const avx_f64pk& a, pat2_<0, 0>)
	{
		return _mm256_movedup_pd(a.v);
	}

	LSIMD_ENSURE_INLINE
	inline avx_f64pk swizzle(const avx_f64pk& a, pat2_<1, 1>)
	{
		return _mm256_unpackhi_pd(a.v, a.v);
	}

	LSIMD_ENSURE_INLINE
	inline avx_f32pk swizzle(const avx_f32pk& a, pat4_<0, 0, 2, 2>)
	{
		return _mm256_moveldup_ps(a.v);
	}

	LSIMD_ENSURE_INLINE
	inline avx_f32pk swizzle(const avx_f32pk& a, pat4_<1, 1, 3, 3>)
	{
		return _mm256_movehdup_ps(a.v);
	}


}

#endif /* AVX_PACK_H_ */
