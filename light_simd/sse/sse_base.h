/**
 * @file sse_base.h
 *
 * @brief The base header file for SSE-based modules
 *
 * This file includes all headers of SSE intrinsics.
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_BASE_H_
#define LSIMD_SSE_BASE_H_

#include <light_simd/common/common_base.h>

#include <xmmintrin.h> 	// for SSE
#include <emmintrin.h> 	// for SSE2

#ifdef LSIMD_HAS_SSE3
#include <pmmintrin.h> 	// for SSE3
#endif

#ifdef LSIMD_HAS_SSSE3
#include <tmmintrin.h> 	// for SSSE3
#endif

#ifdef LSIMD_HAS_SSE4_1
#include <smmintrin.h> 	// for SSE4 (include 4.1 & 4.2)
#endif

#define LSIMD_ALIGN_SSE LSIMD_ALIGN(16)


#endif /* SSE_BASE_H_ */
