/**
 * @file bench_sse_arith.cpp
 *
 * Benchmarking of SSE arithmetic calculation
 *
 * @author Dahua Lin
 */


#include "bench_aux.h"

using namespace lsimd;

const unsigned arr_len = 64;
const unsigned warming_times = 1000;

inline void report_bench(const char *name, unsigned rtimes, uint64_t cycles,
		int pack_w, int nops)
{
	double cpo_f = double(cycles) / (double(rtimes) * double(arr_len));

	int cpoi = int(cpo_f);
	cpoi = (cpoi - nops);  // re-calibrated

	std::printf("\t%-10s:   %4d cycles / %d op\n", name, cpoi, pack_w);
}


template<typename T, template<typename U> class OpT>
inline void bench(unsigned repeat_times, T *pa)
{
	const T lb = OpT<T>::lbound();
	const T ub = OpT<T>::ubound();

	fill_rand(arr_len, pa, lb, ub);

	wrap_op<T, sse_kind, OpT<T>, arr_len> op1(pa);
	uint64_t cs1 = tsc_bench(op1, warming_times, repeat_times);

	report_bench(OpT<T>::name(), repeat_times * OpT<T>::folds(),
			cs1, (int)simd_pack<T, sse_kind>::pack_width, 1);
}


template<typename T>
struct sqrt_op
{
	static const char *name() { return "sqrt"; }

	static T lbound() { return T(0); }
	static T ubound() { return T(100); }

	static unsigned folds() { return 1; }

	LSIMD_ENSURE_INLINE
	static void run(const simd_pack<T, sse_kind>& x)
	{
		simd_pack<T, sse_kind> r = lsimd::sqrt(x);
		force_to_reg(r);
	}
};


template<typename T>
struct rcp_op
{
	static const char *name() { return "rcp"; }

	static T lbound() { return T(1); }
	static T ubound() { return T(100); }

	static unsigned folds() { return 1; }

	LSIMD_ENSURE_INLINE
	static void run(const simd_pack<T, sse_kind>& x)
	{
		simd_pack<T, sse_kind> r = rcp(x);
		force_to_reg(r);
	}
};

template<typename T>
struct rsqrt_op
{
	static const char *name() { return "rsqrt"; }

	static T lbound() { return T(1); }
	static T ubound() { return T(100); }

	static unsigned folds() { return 1; }

	LSIMD_ENSURE_INLINE
	static void run(const simd_pack<T, sse_kind>& x)
	{
		simd_pack<T, sse_kind> r = rsqrt(x);
		force_to_reg(r);
	}
};

template<typename T>
struct approx_rcp_op
{
	static const char *name() { return "rcp(a)"; }

	static T lbound() { return T(1); }
	static T ubound() { return T(100); }

	static unsigned folds() { return 1; }

	LSIMD_ENSURE_INLINE
	static void run(const simd_pack<T, sse_kind>& x)
	{
		simd_pack<T, sse_kind> r = approx_rcp(x.impl);
		force_to_reg(r);
	}
};

template<typename T>
struct approx_rsqrt_op
{
	static const char *name() { return "rsqrt(a)"; }

	static T lbound() { return T(1); }
	static T ubound() { return T(100); }

	static unsigned folds() { return 1; }

	LSIMD_ENSURE_INLINE
	static void run(const simd_pack<T, sse_kind>& x)
	{
		simd_pack<T, sse_kind> r =  approx_rsqrt(x.impl);
		force_to_reg(r);
	}
};

template<typename T>
struct floor_op
{
	static const char *name() { return "floor"; }

	static T lbound() { return T(1); }
	static T ubound() { return T(100); }

	static unsigned folds() { return 1; }

	LSIMD_ENSURE_INLINE
	static void run(const simd_pack<T, sse_kind>& x)
	{
		simd_pack<T, sse_kind> r = floor(x);
		force_to_reg(r);
	}
};

template<typename T>
struct ceil_op
{
	static const char *name() { return "ceil"; }

	static T lbound() { return T(1); }
	static T ubound() { return T(100); }

	static unsigned folds() { return 1; }

	LSIMD_ENSURE_INLINE
	static void run(const simd_pack<T, sse_kind>& x)
	{
		simd_pack<T, sse_kind> r = ceil(x);
		force_to_reg(r);
	}
};

template<typename T>
struct floor2_op
{
	static const char *name() { return "floor(2)"; }

	static T lbound() { return T(1); }
	static T ubound() { return T(100); }

	static unsigned folds() { return 1; }

	LSIMD_ENSURE_INLINE
	static void run(const simd_pack<T, sse_kind>& x)
	{
		simd_pack<T, sse_kind> r = floor_sse2(x.impl);
		force_to_reg(r);
	}
};

template<typename T>
struct ceil2_op
{
	static const char *name() { return "ceil(2)"; }

	static T lbound() { return T(1); }
	static T ubound() { return T(100); }

	static unsigned folds() { return 1; }

	LSIMD_ENSURE_INLINE
	static void run(const simd_pack<T, sse_kind>& x)
	{
		simd_pack<T, sse_kind> r = ceil_sse2(x.impl);
		force_to_reg(r);
	}
};


LSIMD_ALIGN(256) f32 af[arr_len];
LSIMD_ALIGN(256) f32 bf[arr_len];

LSIMD_ALIGN(256) f64 ad[arr_len];
LSIMD_ALIGN(256) f64 bd[arr_len];


#ifdef _MSC_VER
#pragma warning(disable: 4100)
#endif

int main(int argc, char *argv[])
{
	std::printf("Benchmarks on f32\n");
	std::printf("============================\n");

	const unsigned rt_f1 = 2000000;

	bench<f32, sqrt_op>  (rt_f1, af);
	bench<f32, rcp_op>   (rt_f1, af);
	bench<f32, rsqrt_op> (rt_f1, af);

	bench<f32, approx_rcp_op>   (rt_f1, af);
	bench<f32, approx_rsqrt_op> (rt_f1, af);

	bench<f32, floor_op>  (rt_f1, af);
	bench<f32, ceil_op>   (rt_f1, af);
	bench<f32, floor2_op> (rt_f1, af);
	bench<f32, ceil2_op>  (rt_f1, af);

	std::printf("\n");


	std::printf("Benchmarks on f64\n");
	std::printf("============================\n");

	const unsigned rt_d1 = 1000000;

	bench<f64, sqrt_op>  (rt_d1, ad);
	bench<f64, rcp_op>   (rt_d1, ad);
	bench<f64, rsqrt_op> (rt_d1, ad);

	bench<f64, floor_op>  (rt_d1, ad);
	bench<f64, ceil_op>   (rt_d1, ad);
	bench<f64, floor2_op> (rt_d1, ad);
	bench<f64, ceil2_op>  (rt_d1, ad);

	std::printf("\n");

}


