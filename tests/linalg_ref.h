/*
 * @file linalg_ref.h
 *
 * Reference implementation for Linear algebra routines
 *
 * @author Dahua Lin
 */

#ifndef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_LINALG_REF_H_
#define LSIMD_LINALG_REF_H_

#include <light_simd/common/common_base.h>

namespace lsimd
{
	template<typename T, int M, int N>
	struct simple_mat
	{
		T *m_data;

		LSIMD_ENSURE_INLINE simple_mat(T *data) : m_data(data) { }

		LSIMD_ENSURE_INLINE T operator[] (int i) const
		{
			return m_data[i];
		}

		LSIMD_ENSURE_INLINE T& operator[] (int i)
		{
			return m_data[i];
		}

		LSIMD_ENSURE_INLINE T operator() (int i, int j) const
		{
			return m_data[i + j * M];
		}

		LSIMD_ENSURE_INLINE T& operator() (int i, int j)
		{
			return m_data[i + j * M];
		}

		void print(const char *fmt) const
		{
			for (int i = 0; i < M; ++i)
			{
				for (int j = 0; j < N; ++j)
				{
					std::printf(fmt, operator()(i, j));
				}
				std::printf("\n");
			}
		}
	};

	template<typename T, int M, int K, int N>
	inline void ref_mm(
			const simple_mat<T, M, K>& a,
			const simple_mat<T, K, N>& b,
			simple_mat<T, M, N>& c)
	{
		for (int i = 0; i < M * N; ++i) c[i] = T(0);

		for (int j = 0; j < N; ++j)
		{
			for (int k = 0; k < K; ++k)
			{
				for (int i = 0; i < M; ++i)
					c(i, j) += a(i, k) * b(k, j);
			}
		}
	}


}

#endif 
