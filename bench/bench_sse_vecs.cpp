/**
 * @file bench_sse_vecs.cpp
 *
 * Benchmark of SSE vector operations
 *
 * @author Dahua Lin
 */


#include "bench_aux.h"

using namespace lsimd;

const unsigned arr_len = 256;
const unsigned num_vecs = arr_len / 4;
const unsigned warming_times = 10;

LSIMD_ALIGN(128) f32 af[arr_len];
LSIMD_ALIGN(128) f64 ad[arr_len];

LSIMD_ALIGN(128) f32 bf[arr_len];
LSIMD_ALIGN(128) f64 bd[arr_len];

template<typename T> struct data_s;

template<> struct data_s<f32>
{
	LSIMD_ENSURE_INLINE
	static const f32 *src() { return af; }

	LSIMD_ENSURE_INLINE
	static f32 *dst() { return bf; }
};

template<> struct data_s<f64>
{
	LSIMD_ENSURE_INLINE
	static const f64 *src() { return ad; }

	LSIMD_ENSURE_INLINE
	static f64 *dst() { return bd; }
};


template<typename T, int N, template<typename U, int M> class OpT>
inline void bench(unsigned repeat_times)
{
	OpT<T, N> op1;
	uint64_t cs1 = tsc_bench(op1, warming_times, repeat_times);

	double cpv = double(cs1) / (double(repeat_times) * double(num_vecs));

	std::printf("\tf%d x %d:  %.1f cycles / vec\n", (int)(sizeof(T) * 8), N, cpv);
}


template<typename T, int N>
struct addcp_op
{
	const char *name() const { return "add-copy"; }

	LSIMD_ENSURE_INLINE
	void run()
	{
		const T *src = data_s<T>::src();
		T *dst = data_s<T>::dst();

		for (unsigned i = 0; i < num_vecs; ++i)
		{
			simd_vec<T, N, sse_kind> v(src + i * 4, aligned_t());
			simd_vec<T, N, sse_kind> v2(dst + i * 4, aligned_t());

			(v + v2).store(dst + i * 4, aligned_t());
		}
	}
};


template<typename T, int N>
struct ldsum_op
{
	const char *name() const { return "load-sum"; }

	LSIMD_ENSURE_INLINE
	void run()
	{
		const T *src = data_s<T>::src();

		for (unsigned i = 0; i < num_vecs; ++i)
		{
			simd_vec<T, N, sse_kind> v(src + i * 4, aligned_t());

			T s = v.sum();
			force_to_reg(s);
		}
	}
};


template<typename T, int N>
struct lddot_op
{
	const char *name() const { return "load-dot"; }

	LSIMD_ENSURE_INLINE
	void run()
	{
		const T *src = data_s<T>::src();

		for (unsigned i = 0; i < num_vecs; ++i)
		{
			simd_vec<T, N, sse_kind> v(src + i * 4, aligned_t());

			T s = v.dot(v);
			force_to_reg(s);
		}
	}
};



template<template<typename U, int M> class OpT>
void do_bench()
{
	const unsigned int rt_f = 2000000;
	const unsigned int rt_d = rt_f / 2;

	OpT<f32,1> op0;

	std::printf("Benchmarks on %s\n", op0.name());
	std::printf("================================\n");

	bench<f32, 1, OpT>(rt_f);
	bench<f32, 2, OpT>(rt_f);
	bench<f32, 3, OpT>(rt_f);
	bench<f32, 4, OpT>(rt_f);

	std::printf("\t--------------------------\n");

	bench<f64, 1, OpT>(rt_d);
	bench<f64, 2, OpT>(rt_d);
	bench<f64, 3, OpT>(rt_d);
	bench<f64, 4, OpT>(rt_d);

	std::printf("\n");
}


#ifdef _MSC_VER
#pragma warning(disable: 4100)
#endif

int main(int argc, char *argv[])
{
	fill_rand(arr_len, af, 0.f, 1.f);
	fill_rand(arr_len, ad, 0.0, 1.0);

	do_bench<addcp_op>();
	do_bench<ldsum_op>();
	do_bench<lddot_op>();
}




