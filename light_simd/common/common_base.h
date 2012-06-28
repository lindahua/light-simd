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
 *
 * @copyright
 *
 * Copyright (C) 2012 Dahua Lin
 * 
 * Permission is hereby granted, free of charge, to any person 
 * obtaining a copy of this software and associated documentation 
 * files (the "Software"), to deal in the Software without restriction, 
 * including without limitation the rights to use, copy, modify, merge, 
 * publish, distribute, sublicense, and/or sell copies of the Software, 
 * and to permit persons to whom the Software is furnished to do so, 
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be 
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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



/**
 * @defgroup core_module Core Module
 *
 * @brief  SIMD pack classes, arithmetic functions, 
 *         and supporting facilities.
 *
 * The core module comprises a set of SIMD pack classes that 
 * serve as the core of the entire library, which particularly
 * include
 *
 * - A generic SIMD pack class: lsimd::simd_pack
 * - A set of specific SIMD pack classes:
 *   - lsimd::sse_pack<f32>
 *   - lsimd::sse_pack<f64>
 *
 * In addition, it also contains 
 * - A set of useful supporting facilities (see \ref common_base.h)
 * - Overloaded operators and arithmetic functions 
 *   (see \ref simd_arith.h and \ref sse_arith.h)
 */



/**
 * The main namespace of the Light SIMD library.
 *
 * Most names, including classes, functions, constants
 * are declared in this namespace.
 */
namespace lsimd
{

	/**
	 * @defgroup basic_defs Basic Definitions
	 * @ingroup core_module
	 *
	 * @brief Definitions of basic types used by the 
	 *        entire library.
	 */

	/**
	 * @defgroup scalar_types Scalar Types
	 * @ingroup basic_defs
	 *
	 * @brief The primitive scalar types to represent a single 
	 *        scalar. 
	 *
	 * The primitive types are mostly typedefs of builtin types, using 
	 * consistent naming.
	 */ 
	/** @{ */ 

	/** 
	 * @brief 8-bit signed integer. 
	 */
	typedef  int8_t  i8;

	/**
	 * @brief 8-bit unsigned integer.
	 */
	typedef uint8_t  u8;

	/**
	 * @brief 16-bit signed integer.
	 */
	typedef  int16_t i16;

	/**
	 * @brief 16-bit unsigned integer.
	 */
	typedef uint16_t u16;

	/**
	 * @brief 32-bit signed integer.
	 */
	typedef  int32_t i32;

	/**
	 * @brief 32-bit unsigned integer.
	 */
	typedef uint32_t u32;

	/**
	 * @brief Single-precision (32-bit) floating-point real number.
	 */
	typedef float  f32;

	/**
	 * @brief Double-precision (64-bit) floating-point real number.
	 */
	typedef double f64;

	/**
	 * @brief The unsigned integer type to represent sizes.
	 *
	 * @remark This is simply using std::size_t.
	 */
	using std::size_t;

	/**
	 * @brief The unsigned integer type to represent the offsets.
	 *
	 * @remark This is simply using std::ptrdiff_t.
	 */
	using std::ptrdiff_t;

	typedef i32 index_t;

	/** @} */ // scalar_types


	/**
	 * @defgroup tag_types Tag Types
	 * @ingroup basic_defs
	 *
	 * @brief The tag types for meta-programming.
	 *
	 * The tag types are mostly empty struct types to indicate a particular
	 * operation or attribute at compile-time.
	 */ 
	/** @{ */ 

	/**
	 * @brief tag type: memory addresses are properly aligned.
	 *
	 * @see unaligned_t.
	 */
	struct aligned_t { };

	/**
	 * @brief tag type: memory addresses are not necessarily aligned.
	 *
	 * @see aligned_t
	 */
	struct unaligned_t { };

	/**
	 * @brief tag type: initialize all elements to zero values.
	 */
	struct zero_t { };


	/**
	 * @brief tag type: use SSE for SIMD computation.
	 */
	struct sse_kind { };

	/**
	 * @brief The default kind of data types for SIMD processing.
	 *
	 * @remark Currently, only SSE has been supported by the library,
	 *         and thus this is set to be the same as \ref sse_kind.
	 *         It will be set depending on platform, when more
	 *         SIMD kinds (e.g. AVX and NEON) are supported.
	 *
	 * @see sse_kind.
	 */
	typedef sse_kind default_simd_kind;

	/** @} */  // tag_types


	// forward declarations

	template<typename T, typename Kind> struct simd;

	template<typename T, typename Kind=default_simd_kind> struct simd_pack;

	template<typename T, int N, typename Kind=default_simd_kind> struct simd_vec;

	template<typename T, int M, int N, typename Kind=default_simd_kind> struct simd_mat;

}

/**
 * @ingroup basic_defs
 * @{
 */

#if (LSIMD_COMPILER == LSIMD_GCC || LSIMD_COMPILER == LSIMD_CLANG )

/**
 * Specifies the minimum alignment of the ensuring variable/array.
 *
 * For example, the following code
 *
 * \code{.cpp}
 * LSIMD_ALIGN(16) a[4];
 * \endcode
 *
 * ensures that the array a is aligned to 16-byte boundary
 */
#define LSIMD_ALIGN(n) __attribute__((aligned(n)))

/**
 * Forces the ensuing function to be inlined.
 */
#define LSIMD_ENSURE_INLINE __attribute__((always_inline))

#elif (LSIMD_COMPILER == LSIMD_MSVC)

#define LSIMD_ALIGN(n) __declspec(align(n))
#define LSIMD_ENSURE_INLINE __forceinline

#endif

/** @} */


#endif /* COMMON_BASE_H_ */
