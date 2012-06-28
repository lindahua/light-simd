/*
 * @file bench_sse_mm.cpp
 *
 * Benchmarking of SSE-based matrix product
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

LSIMD_ALIGN(128) f32 cf[arr_len];
LSIMD_ALIGN(128) f64 cd[arr_len];


template<typename T> struct data_s;

template<> struct data_s<f32>
{
	LSIMD_ENSURE_INLINE
	static const f32 *a() { return af; }

	LSIMD_ENSURE_INLINE
	static const f32 *b() { return bf; }

	LSIMD_ENSURE_INLINE
	static f32 *c() { return cf; }
};

template<> struct data_s<f64>
{
	LSIMD_ENSURE_INLINE
	static const f64 *a() { return ad; }

	LSIMD_ENSURE_INLINE
	static const f64 *b() { return bd; }

	LSIMD_ENSURE_INLINE
	static f64 *c() { return cd; }
};


template<typename T, int M, int K, int N>
struct mtimes_cp
{
	LSIMD_ENSURE_INLINE
	void run()
	{
		const T *pa = data_s<T>::a();
		const T *pb = data_s<T>::b();
		T *pc = data_s<T>::c();

		simd_mat<T, M, K, sse_kind> a;
		simd_mat<T, K, N, sse_kind> b;
		simd_mat<T, M, N, sse_kind> c;

		for (unsigned i = 0; i < num_mats; ++i)
		{
			a.load(pa + i * step_size, aligned_t());
			b.load(pb + i * step_size, aligned_t());

			c = a * b;

			c.store(pc + i * step_size, aligned_t());
		}
	}
};


template<typename T, int M, int K, int N>
inline void bench(unsigned repeat_times)
{
	mtimes_cp<T, M, K, N> op1;
	uint64_t cs1 = tsc_bench(op1, warming_times, repeat_times);

	double cpv = double(cs1) / (double(repeat_times) * double(num_mats));

	std::printf("(%d x %d) * (%d x %d):  %6.1f cycles / mat ==> %.1f scalar-op / cycle\n",
			M, K, K, N, cpv, double(2 * M * K * N) / cpv);
}


template<typename T>
void do_bench(const unsigned int rt)
{
	std::printf("Benchmarks on f%d\n", int(8 * sizeof(T)));
	std::printf("================================\n");

	bench<T, 2, 2, 2>(rt);
	bench<T, 2, 2, 3>(rt);
	bench<T, 2, 2, 4>(rt);

	bench<T, 2, 3, 2>(rt);
	bench<T, 2, 3, 3>(rt);
	bench<T, 2, 3, 4>(rt);

	bench<T, 2, 4, 2>(rt);
	bench<T, 2, 4, 3>(rt);
	bench<T, 2, 4, 4>(rt);

	bench<T, 3, 2, 2>(rt);
	bench<T, 3, 2, 3>(rt);
	bench<T, 3, 2, 4>(rt);

	bench<T, 3, 3, 2>(rt);
	bench<T, 3, 3, 3>(rt);
	bench<T, 3, 3, 4>(rt);

	bench<T, 3, 4, 2>(rt);
	bench<T, 3, 4, 3>(rt);
	bench<T, 3, 4, 4>(rt);

	bench<T, 4, 2, 2>(rt);
	bench<T, 4, 2, 3>(rt);
	bench<T, 4, 2, 4>(rt);

	bench<T, 4, 3, 2>(rt);
	bench<T, 4, 3, 3>(rt);
	bench<T, 4, 3, 4>(rt);

	bench<T, 4, 4, 2>(rt);
	bench<T, 4, 4, 3>(rt);
	bench<T, 4, 4, 4>(rt);

	std::printf("\n");
}

#ifdef _MSC_VER
#pragma warning(disable: 4100)
#endif

int main(int argc, char *argv[])
{
	fill_rand(arr_len, af, 0.f, 1.f);
	fill_rand(arr_len, ad, 0.0, 1.0);
	fill_rand(arr_len, bf, 0.f, 1.f);
	fill_rand(arr_len, bd, 0.0, 1.0);

	const unsigned int rt_f = 2000000;
	const unsigned int rt_d = rt_f / 2;

	do_bench<f32>(rt_f);
	do_bench<f64>(rt_d);
}







