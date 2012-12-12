/**
 * @file common_base.h
 *
 * @brief A set of basic types and macros that will be used by the entire library.
 *
 * This file, particularly, contains the definitions of
 *
 * - Basic numeric types (e.g. f32, f64, i32, etc)
 * - Basic tag types (e.g. aligned_t, unaligned_t and zero_t)
 * - Forward declaration of basic template classes (e.g. simd_pack, simd_vec, etc)
 * - A set of useful macros
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_COMMON_BASE_H_
#define LSIMD_COMMON_BASE_H_

#include <light_simd/arch.h>

#include <cstddef>
#include <stdint.h>
#include <cstdio>


#if (LSIMD_COMPILER == LSIMD_GCC || LSIMD_COMPILER == LSIMD_CLANG )

#define LSIMD_ALIGN(n) __attribute__((aligned(n)))
#define LSIMD_ENSURE_INLINE __attribute__((always_inline))

#elif (LSIMD_COMPILER == LSIMD_MSVC)

#define LSIMD_ALIGN(n) __declspec(align(n))
#define LSIMD_ENSURE_INLINE __forceinline

#endif

namespace lsimd
{
	// primitive types

	typedef  int8_t  i8;
	typedef uint8_t  u8;
	typedef  int16_t i16;
	typedef uint16_t u16;
	typedef  int32_t i32;
	typedef uint32_t u32;

	typedef float  f32;
	typedef double f64;

	using std::size_t;
	using std::ptrdiff_t;

	typedef i32 index_t;

	// tag types

	namespace tag
	{
		struct aligned { };
		struct unaligned { };

		struct all_zeros { };
		struct all_nonzeros { };
	}

	// SIMD kind tags

	struct sse_kind { };
	struct avx_kind { };

	typedef sse_kind default_simd_kind;

	// Auxiliary types

	template<int I>
	struct int_
	{
		static const int value = I;
	};

	template<int I0, int I1>
	struct pat2_{ };

	template<int I0, int I1, int I2, int I3>
	struct pat4_{ };

	template<int I0, int I1, int I2, int I3, int I4, int I5, int I6, int I7>
	struct pat8_{ };


	// forward declarations

	template<typename T, typename Kind> struct simd_traits;

	template<typename T, typename Kind=default_simd_kind> struct simd_pack;

	template<typename T, int N, typename Kind=default_simd_kind> struct simd_vec;

	template<typename T, int M, int N, typename Kind=default_simd_kind> struct simd_mat;

}



#endif /* COMMON_BASE_H_ */


