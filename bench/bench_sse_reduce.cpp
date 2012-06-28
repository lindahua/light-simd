/**
 * @file bench_sse_reduce.cpp
 *
 * Benchmarking for SSE-based vector reduction
 *
 * @author Dahua Lin
 */


#include "bench_aux.h"

using namespace lsimd;

const unsigned arr_len = 128;
const unsigned warming_times = 10;

LSIMD_ALIGN(128) f32 af[arr_len];
LSIMD_ALIGN(128) f64 ad[arr_len];

inline void report_bench(const char *name, unsigned rtimes, uint64_t cycles,
		int pack_w, int nops)
{
	double cpo_f = double(cycles) / (double(rtimes) * double(arr_len));

	int cpoi = int(cpo_f);
	cpoi = (cpoi - nops);  // re-calibrated

	std::printf("\t%-8s:   %4d cycles / %d op\n", name, cpoi, pack_w);
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
struct sum_op
{
	static const char *name() { return "sum"; }

	static T lbound() { return T(0); }
	static T ubound() { return T(100); }

	static unsigned folds() { return 1; }

	LSIMD_ENSURE_INLINE
	static void run(simd_pack<T, sse_kind> x)
	{
		T r = x.sum();
		force_to_reg(r);
	}
};


template<typename T>
struct psum2_op
{
	static const char *name() { return "psum2"; }

	static T lbound() { return T(0); }
	static T ubound() { return T(100); }

	static unsigned folds() { return 1; }

	LSIMD_ENSURE_INLINE
	static void run(const simd_pack<f32, sse_kind>& x)
	{
		T r = x.partial_sum<2>();
		force_to_reg(r);
	}
};


template<typename T>
struct psum3_op
{
	static const char *name() { return "psum3"; }

	static T lbound() { return T(0); }
	static T ubound() { return T(100); }

	static unsigned folds() { return 1; }

	LSIMD_ENSURE_INLINE
	static void run(const simd_pack<f32, sse_kind>& x)
	{
		T r = x.partial_sum<3>();
		force_to_reg(r);
	}
};


template<typename T>
struct max_op
{
	static const char *name() { return "max"; }

	static T lbound() { return T(0); }
	static T ubound() { return T(100); }

	static unsigned folds() { return 1; }

	LSIMD_ENSURE_INLINE
	static void run(const simd_pack<T, sse_kind>& x)
	{
		T r = x.max();
		force_to_reg(r);
	}
};

template<typename T>
struct pmax2_op
{
	static const char *name() { return "pmax2"; }

	static T lbound() { return T(0); }
	static T ubound() { return T(100); }

	static unsigned folds() { return 1; }

	LSIMD_ENSURE_INLINE
	static void run(const simd_pack<f32, sse_kind>& x)
	{
		T r = x.partial_max<2>();
		force_to_reg(r);
	}
};


template<typename T>
struct pmax3_op
{
	static const char *name() { return "pmax3"; }

	static T lbound() { return T(0); }
	static T ubound() { return T(100); }

	static unsigned folds() { return 1; }

	LSIMD_ENSURE_INLINE
	static void run(const simd_pack<f32, sse_kind>& x)
	{
		T r = x.partial_max<3>();
		force_to_reg(r);
	}
};


template<typename T>
struct min_op
{
	static const char *name() { return "min"; }

	static T lbound() { return T(0); }
	static T ubound() { return T(100); }

	static unsigned folds() { return 1; }

	LSIMD_ENSURE_INLINE
	static void run(const simd_pack<T, sse_kind>& x)
	{
		T r = x.min();
		force_to_reg(r);
	}
};


template<typename T>
struct pmin2_op
{
	static const char *name() { return "pmin2"; }

	static T lbound() { return T(0); }
	static T ubound() { return T(100); }

	static unsigned folds() { return 1; }

	LSIMD_ENSURE_INLINE
	static void run(const simd_pack<f32, sse_kind>& x)
	{
		T r = x.partial_min<2>();
		force_to_reg(r);
	}
};


template<typename T>
struct pmin3_op
{
	static const char *name() { return "pmin3"; }

	static T lbound() { return T(0); }
	static T ubound() { return T(100); }

	static unsigned folds() { return 1; }

	LSIMD_ENSURE_INLINE
	static void run(const simd_pack<f32, sse_kind>& x)
	{
		T r = x.partial_min<3>();
		force_to_reg(r);
	}
};

#ifdef _MSC_VER
#pragma warning(disable: 4100)
#endif

int main(int argc, char *argv[])
{
	std::printf("Benchmarks on f32\n");
	std::printf("============================\n");

	const unsigned rt_f1 = 2000000;

	bench<f32, sum_op>  (rt_f1, af);
	bench<f32, min_op>  (rt_f1, af);
	bench<f32, max_op>  (rt_f1, af);

	bench<f32, psum2_op> (rt_f1, af);
	bench<f32, psum3_op> (rt_f1, af);
	bench<f32, pmax2_op> (rt_f1, af);
	bench<f32, pmax3_op> (rt_f1, af);
	bench<f32, pmin2_op> (rt_f1, af);
	bench<f32, pmin3_op> (rt_f1, af);

	std::printf("\n");

	std::printf("Benchmarks on f64\n");
	std::printf("============================\n");

	const unsigned rt_d1 = 1000000;

	bench<f64, sum_op>  (rt_d1, ad);
	bench<f64, min_op>  (rt_d1, ad);
	bench<f64, max_op>  (rt_d1, ad);

	std::printf("\n");
}





