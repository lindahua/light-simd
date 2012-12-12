/**
 * @file avx_base.h
 *
 * @brief The base header file for AVX-based modules
 * 
 * @author Dahua Lin 
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef AVX_BASE_H_
#define AVX_BASE_H_

#include <light_simd/common/common_base.h>

#ifdef LSIMD_HAS_AVX
#include <immintrin.h>
#endif

#define LSIMD_ALIGN_AVX LSIMD_ALIGN(32)

#endif 
