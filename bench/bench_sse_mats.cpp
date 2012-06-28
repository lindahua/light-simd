/**
 * @file bench_sse_mats.cpp
 *
 * Benchmark of SSE matrix operations
 *
 * @author Dahua Lin
 */


#include "bench_aux.h"

using namespace lsimd;

const unsigned arr_len = 512;
const unsigned step_size = 16;
const unsigned num_mats = arr_len / step_size;
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


template<typename T, int M, int N, template<typename U, int M_, int N_> class OpT>
inline void bench(unsigned repeat_times)
{
	OpT<T, M, N> op1;
	uint64_t cs1 = tsc_bench(op1, warming_times, repeat_times);

	double cpv = double(cs1) / (double(repeat_times) * double(num_mats));

	std::printf("\tf%d %d x %d:  %6.1f cycles / mat ==> %.1f scalar-op / cycle\n",
			(int)(sizeof(T) * 8), M, N, cpv, op1.scalar_ops() / cpv);
}

template<typename T, int N, template<typename U, int N_> class OpT>
inline void bench1(unsigned repeat_times)
{
	OpT<T, N> op1;
	uint64_t cs1 = tsc_bench(op1, warming_times, repeat_times);

	double cpv = double(cs1) / (double(repeat_times) * double(num_mats));

	std::printf("\tf%d %d x %d:  %6.1f cycles / mat ==> %.1f scalar-op / cycle\n",
			(int)(sizeof(T) * 8), N, N, cpv, op1.scalar_ops() / cpv);
}



template<typename T, int M, int N>
struct addcp_op
{
	const char *name() const { return "add-copy"; }

	int scalar_ops() const { return M * N; }

	LSIMD_ENSURE_INLINE
	void run()
	{
		const T *src = data_s<T>::src();
		T *dst = data_s<T>::dst();

		for (unsigned i = 0; i < num_mats; ++i)
		{
			simd_mat<T, M, N, sse_kind> a(src + i * step_size, aligned_t());
			simd_mat<T, M, N, sse_kind> b(dst + i * step_size, aligned_t());

			(a + b).store(dst + i * step_size, aligned_t());
		}
	}
};


template<typename T, int M, int N>
struct addip_op
{
	const char *name() const { return "add-update"; }

	int scalar_ops() const { return M * N; }

	LSIMD_ENSURE_INLINE
	void run()
	{
		const T *src = data_s<T>::src();
		T *dst = data_s<T>::dst();

		for (unsigned i = 0; i < num_mats; ++i)
		{
			simd_mat<T, M, N, sse_kind> a(src + i * step_size, aligned_t());
			simd_mat<T, M, N, sse_kind> b(dst + i * step_size, aligned_t());

			a += b;

			a.store(dst + i * step_size, aligned_t());
		}
	}
};



template<typename T, int M, int N>
struct transcp_op
{
	const char *name() const { return "trans-copy"; }

	int scalar_ops() const { return M * N; }

	LSIMD_ENSURE_INLINE
	void run()
	{
		const T *src = data_s<T>::src();
		T *dst = data_s<T>::dst();

		for (unsigned i = 0; i < num_mats; ++i)
		{
			simd_mat<T, M, N, sse_kind> a;
			a.load_trans(src + i * step_size, aligned_t());
			a.store(dst + i * step_size, aligned_t());
		}
	}
};


template<typename T, int M, int N>
struct mtimes_op
{
	const char *name() const { return "mtimes-copy"; }

	int scalar_ops() const { return 2 * M * N; }

	LSIMD_ENSURE_INLINE
	void run()
	{
		const T *src = data_s<T>::src();
		T *dst = data_s<T>::dst();

		simd_mat<T, M, N, sse_kind> a;
		a.load(src, aligned_t());

		for (unsigned i = 0; i < num_mats; ++i)
		{
			simd_vec<T, N, sse_kind> v;
			v.load(src + i * step_size, aligned_t());

			(a * v).store(dst + i * step_size, aligned_t());
		}
	}
};


template<typename T, int N>
struct inv_op
{
	const char *name() const { return "inv-copy"; }

	int scalar_ops() const
	{
		return 4 * N * N * N;
	}

	LSIMD_ENSURE_INLINE
	void run()
	{
		const T *src = data_s<T>::src();
		T *dst = data_s<T>::dst();

		simd_mat<T, N, N, sse_kind> a, b;

		for (unsigned i = 0; i < num_mats; ++i)
		{
			a.load(src + i * step_size, aligned_t());
			inv_and_det(a, b);
			b.store(dst + i * step_size, aligned_t());
		}
	}
};


template<typename T, int N>
struct solve_op
{
	const char *name() const { return "solve-copy"; }

	int scalar_ops() const { return 4 * N * N * N; }

	LSIMD_ENSURE_INLINE
	void run()
	{
		const T *src = data_s<T>::src();
		T *dst = data_s<T>::dst();

		simd_mat<T, N, N, sse_kind> a;
		a.load(src, aligned_t());

		for (unsigned i = 0; i < num_mats; ++i)
		{
			simd_vec<T, N, sse_kind> v;
			v.load(src + i * step_size, aligned_t());

			solve(a, v).store(dst + i * step_size, aligned_t());
		}
	}
};



template<template<typename U, int M, int N> class OpT>
void do_bench()
{
	const unsigned int rt_f = 2000000;
	const unsigned int rt_d = rt_f / 2;

	OpT<f32,2,2> op0;

	std::printf("Benchmarks on %s\n", op0.name());
	std::printf("================================\n");

	bench<f32, 2, 2, OpT>(rt_f);
	bench<f32, 2, 3, OpT>(rt_f);
	bench<f32, 2, 4, OpT>(rt_f);
	bench<f32, 3, 2, OpT>(rt_f);
	bench<f32, 3, 3, OpT>(rt_f);
	bench<f32, 3, 4, OpT>(rt_f);
	bench<f32, 4, 2, OpT>(rt_f);
	bench<f32, 4, 3, OpT>(rt_f);
	bench<f32, 4, 4, OpT>(rt_f);

	std::printf("\t-------------------------------------------------------\n");

	bench<f64, 2, 2, OpT>(rt_d);
	bench<f64, 2, 3, OpT>(rt_d);
	bench<f64, 2, 4, OpT>(rt_d);
	bench<f64, 3, 2, OpT>(rt_d);
	bench<f64, 3, 3, OpT>(rt_d);
	bench<f64, 3, 4, OpT>(rt_d);
	bench<f64, 4, 2, OpT>(rt_d);
	bench<f64, 4, 3, OpT>(rt_d);
	bench<f64, 4, 4, OpT>(rt_d);

	std::printf("\n");
}


template<template<typename U, int N> class OpT>
void do_bench1()
{
	const unsigned int rt_f = 2000000;
	const unsigned int rt_d = rt_f / 2;

	OpT<f32,2> op0;

	std::printf("Benchmarks on %s\n", op0.name());
	std::printf("================================\n");

	bench1<f32, 2, OpT>(rt_f);
	bench1<f32, 3, OpT>(rt_f);
	bench1<f32, 4, OpT>(rt_f);

	std::printf("\t-------------------------------------------------------\n");

	bench1<f64, 2, OpT>(rt_d);
	bench1<f64, 3, OpT>(rt_d);
	bench1<f64, 4, OpT>(rt_d);

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
	do_bench<addip_op>();
	do_bench<transcp_op>();
	do_bench<mtimes_op>();

	do_bench1<inv_op>();
	do_bench1<solve_op>();
}








