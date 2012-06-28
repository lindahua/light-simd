/**
 * @file arch.h
 *
 * The macros for architecture detection
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LIGHTSIMD_ARCH_H_
#define LIGHTSIMD_ARCH_H_

// Compiler detection


#define LSIMD_MSVC  0x01
#define LSIMD_GCC   0x02
#define LSIMD_CLANG 0x03


#if (defined(_WIN32) || defined(_WIN64)) && defined(_MSC_VER)
	#if _MSC_VER < 1600
		#error Microsoft Visual C++ of version lower than MSVC 2010 is not supported.
	#endif
	#define LSIMD_COMPILER LSIMD_MSVC

#elif (defined(__GNUC__))

	#if (defined(__clang__))
		#if ((__clang_major__ < 2) || (__clang_major__ == 2 && __clang_minor__ < 8))
			#error CLANG of version lower than 2.8.0 is not supported
		#endif
		#define LSIMD_COMPILER LSIMD_CLANG

	#else
		#if ((__GNUC__ < 4) || (__GNUC__ == 4 && __GNUC_MINOR__ < 2))
			#error GCC of version lower than 4.2.0 is not supported
		#endif
		#define LSIMD_COMPILER LSIMD_GCC
	#endif

#else
	#error Light-SIMD can only be used with Microsoft Visual C++, GCC (G++), or clang (clang++).
#endif


#ifdef __GXX_EXPERIMENTAL_CXX0X__
#define LSIMD_HAS_C99_SCALAR_MATH
#endif


// SIMD support detection

#if defined(_MSC_VER)

#if _M_IX86_FP >= 2
#define LSIMD_HAS_SSE2
#endif

#else

#if defined(__SSE2__)
#define LSIMD_HAS_SSE2
#endif

#if defined(__SSE3__)
#define LSIMD_HAS_SSE3
#endif

#if defined(__SSSE3__)
#define LSIMD_HAS_SSSE3
#endif

#if defined(__SSE4_1__)
#define LSIMD_HAS_SSE4_1
#endif

#if defined(__SSE4_2__)
#define LSIMD_HAS_SSE4_2
#endif

#endif

#ifndef LSIMD_HAS_SSE2
	#error Light-SIMD needs SSE2 support to work.
#endif

#endif




