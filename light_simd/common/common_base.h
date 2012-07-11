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

	struct aligned_t { };
	struct unaligned_t { };

	struct zero_t { };

	// SIMD kind tags

	struct sse_kind { };

	typedef sse_kind default_simd_kind;


	// forward declarations

	template<typename T, typename Kind> struct simd;

	template<typename T, typename Kind=default_simd_kind> struct simd_pack;

	template<typename T, int N, typename Kind=default_simd_kind> struct simd_vec;

	template<typename T, int M, int N, typename Kind=default_simd_kind> struct simd_mat;

}

#if (LSIMD_COMPILER == LSIMD_GCC || LSIMD_COMPILER == LSIMD_CLANG )

#define LSIMD_ALIGN(n) __attribute__((aligned(n)))
#define LSIMD_ENSURE_INLINE __attribute__((always_inline))

#elif (LSIMD_COMPILER == LSIMD_MSVC)

#define LSIMD_ALIGN(n) __declspec(align(n))
#define LSIMD_ENSURE_INLINE __forceinline

#endif

#endif /* COMMON_BASE_H_ */


