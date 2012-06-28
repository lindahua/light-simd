/**
 * @file bench_aux.h
 *
 * Auxiliary facilities for benchmarking
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_BENCH_AUX_H_
#define LSIMD_BENCH_AUX_H_

#include <light_simd/simd.h>
#include <cmath>
#include <cstdlib>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4141)
#define ASM __asm

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#endif

namespace lsimd
{
	/********************************************
	 *
	 *  Timing
	 *
	 ********************************************/

#ifdef _MSC_VER
	
#error Performance counter is not yet ready for MSVC

#else
	LSIMD_ENSURE_INLINE
	inline uint64_t read_tsc(void) 
	{
	    uint32_t lo, hi;
	    __asm__ volatile("rdtscp" : "=a"(lo), "=d"(hi) :: "ecx" );
	    return (uint64_t)hi << 32 | lo;
	}

	template<typename T>
	LSIMD_ENSURE_INLINE
	inline void force_to_reg(const simd_pack<T, sse_kind>& x)
	{
		__asm__ volatile("" : : "x"(x.impl.v));
	}

	LSIMD_ENSURE_INLINE
	inline void force_to_reg(const f32& x)
	{
		__asm__ volatile("" : : "x"(x));
	}

	LSIMD_ENSURE_INLINE
	inline void force_to_reg(const f64& x)
	{
		__asm__ volatile("" : : "x"(x));
	}

#endif

	template<class Op>
	uint64_t tsc_bench(Op op, unsigned warming_times, unsigned repeat_times)
	{
		for (unsigned i = 0; i < warming_times; ++i) op.run();

		uint64_t tic = read_tsc();
		for (unsigned i = 0; i < repeat_times; ++i) op.run();
		uint64_t toc = read_tsc();

		return toc - tic;  // total cycles
	}


	template<typename T, typename Kind, class Op, unsigned Len>
	struct wrap_op
	{
		static const unsigned w = simd<T, Kind>::pack_width;
		const T *a;

		wrap_op(const T *a_)
		: a(a_)
		{
		}

		LSIMD_ENSURE_INLINE
		void run()
		{
			for (unsigned i = 0; i < Len; ++i)
			{
				simd_pack<T, Kind> x0(a + i * w, aligned_t());
				force_to_reg(x0);

				Op::run(x0);
			}
		}
	};

	template<typename T, typename Kind, class Op, unsigned Len>
	struct wrap_op2
	{
		static const unsigned w = simd<T, Kind>::pack_width;
		const T *a;
		const T *b;

		wrap_op2(const T *a_, const T *b_)
		: a(a_), b(b_)
		{
		}

		LSIMD_ENSURE_INLINE
		void run()
		{
			for (unsigned i = 0; i < Len; ++i)
			{
				simd_pack<T, Kind> x0(a + i * w, b + i * w, aligned_t());
				force_to_reg(x0);

				simd_pack<T, Kind> y0(a + i * w, b + i * w, aligned_t());
				force_to_reg(y0);

				Op::run(x0, y0);
			}
		}
	};


	/********************************************
	 *
	 *  Array functions
	 *
	 ********************************************/


	template<typename T>
	inline void clear_zeros(int n, T *a)
	{
		for (int i = 0; i < n; ++i) a[i] = T(0);
	}

	template<typename T>
	inline void fill_const(int n, T *a, T v)
	{
		for (int i = 0; i < n; ++i) a[i] = v;
	}


	template<typename T>
	inline T rand_val(const T lb, const T ub)
	{
		double r = double(std::rand()) / RAND_MAX;
		r = double(lb) + r * double(ub - lb);
		return T(r);
	}

	template<typename T>
	inline void fill_rand(int n, T *a, T lb, T ub)
	{
		for (int i = 0; i < n; ++i)
		{
			a[i] = rand_val(lb, ub);
		}
	}

}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif /* BENCH_AUX_H_ */
