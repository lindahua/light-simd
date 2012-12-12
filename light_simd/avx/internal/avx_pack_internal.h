/**
 * @file avx_pack_internal.h
 *
 * @brief Internal implementation of part of AVX packs
 *
 * @author Dahua Lin
 */

#ifndef LSIMD_AVX_PACK_INTERNAL_H_
#define LSIMD_AVX_PACK_INTERNAL_H_

#include <light_simd/avx/avx_base.h>

namespace lsimd { namespace avx_internal {

	LSIMD_ENSURE_INLINE
	inline __m256i partial_mask_i32(int_<0>)
	{
		return _mm256_setzero_si256();
	}

	LSIMD_ENSURE_INLINE
	inline __m256i partial_mask_i32(int_<1>)
	{
		return _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0,
				(int)0xffffffff);
	}

	LSIMD_ENSURE_INLINE
	inline __m256i partial_mask_i32(int_<2>)
	{
		return _mm256_set_epi32(0, 0, 0, 0, 0, 0,
				(int)0xffffffff,
				(int)0xffffffff);
	}

	LSIMD_ENSURE_INLINE
	inline __m256i partial_mask_i32(int_<3>)
	{
		return _mm256_set_epi32(0, 0, 0, 0, 0,
				(int)0xffffffff,
				(int)0xffffffff,
				(int)0xffffffff);
	}


	LSIMD_ENSURE_INLINE
	inline __m256i partial_mask_i32(int_<4>)
	{
		return _mm256_set_epi32(0, 0, 0, 0,
				(int)0xffffffff,
				(int)0xffffffff,
				(int)0xffffffff,
				(int)0xffffffff);
	}

	LSIMD_ENSURE_INLINE
	inline __m256i partial_mask_i32(int_<5>)
	{
		return _mm256_set_epi32(0, 0, 0,
				(int)0xffffffff,
				(int)0xffffffff,
				(int)0xffffffff,
				(int)0xffffffff,
				(int)0xffffffff);
	}

	LSIMD_ENSURE_INLINE
	inline __m256i partial_mask_i32(int_<6>)
	{
		return _mm256_set_epi32(0, 0,
				(int)0xffffffff,
				(int)0xffffffff,
				(int)0xffffffff,
				(int)0xffffffff,
				(int)0xffffffff,
				(int)0xffffffff);
	}

	LSIMD_ENSURE_INLINE
	inline __m256i partial_mask_i32(int_<7>)
	{
		return _mm256_set_epi32(0,
				(int)0xffffffff,
				(int)0xffffffff,
				(int)0xffffffff,
				(int)0xffffffff,
				(int)0xffffffff,
				(int)0xffffffff,
				(int)0xffffffff);
	}

	LSIMD_ENSURE_INLINE
	inline __m256i partial_mask_i32(int_<8>)
	{
		return _mm256_set_epi32(
				(int)0xffffffff,
				(int)0xffffffff,
				(int)0xffffffff,
				(int)0xffffffff,
				(int)0xffffffff,
				(int)0xffffffff,
				(int)0xffffffff,
				(int)0xffffffff);
	}



} }


#endif /* AVX_PACK_INTERNAL_H_ */
