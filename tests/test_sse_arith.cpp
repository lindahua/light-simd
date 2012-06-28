/**
 * @file test_sse_arith.cpp
 *
 * Test the accuracy of Arithmetic functions
 *
 * @author Dahua Lin
 */

#include "test_aux.h"
#include <light_test/color_printf.h>

using namespace lsimd;

const int N = 100 * 1024;

const double tol_f32 = 1.e-6;
const double tol_f64 = 1.0e-15;

inline void print_pass(bool passed)
{
	if (passed)
		ltest::printf_with_color(ltest::LTCOLOR_GREEN, "passed");
	else
		ltest::printf_with_color(ltest::LTCOLOR_RED, "failed");
}


template<typename T, template<typename U> class OpT>
bool test_accuracy_u(T tol = sizeof(T) == 4 ? T(tol_f32) : T(tol_f64))
{
	T lb_x = OpT<T>::lb_x();
	T ub_x = OpT<T>::ub_x();

	double maxdev = eval_approx_accuracy<T, sse_kind, OpT<T> >(N, lb_x, ub_x);
	bool passed = maxdev < tol;

	std::printf("\t%-9s:    max-rdev = %10.3g  ... ", OpT<T>::name(), maxdev);
	print_pass(passed);
	std::printf("\n");

	return passed;
}

template<typename T, template<typename U> class OpT>
bool test_accuracy_b(T tol = sizeof(T) == 4 ? T(tol_f32) : T(tol_f64))
{
	T lb_x = OpT<T>::lb_x();
	T ub_x = OpT<T>::ub_x();

	T lb_y = OpT<T>::lb_y();
	T ub_y = OpT<T>::ub_y();

	double maxdev = eval_approx_accuracy<T, sse_kind, OpT<T> >(N, lb_x, ub_x, lb_y, ub_y);
	bool passed = maxdev < tol;

	std::printf("\t%-9s:    max-rdev = %10.3g  ... ", OpT<T>::name(), maxdev);
	print_pass(passed);
	std::printf("\n");

	return passed;
}


template<typename T>
struct add_ts
{
	static const char *name() { return "add"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T lb_y() { return T(-10); }
	static T ub_y() { return T(10); }

	static T eval_scalar(const T x, const T y) { return x + y; }

	static simd_pack<T, sse_kind> eval_vector(const simd_pack<T, sse_kind> x, const simd_pack<T, sse_kind> y)
	{
		return x + y;
	}
};


template<typename T>
struct sub_ts
{
	static const char *name() { return "sub"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T lb_y() { return T(-10); }
	static T ub_y() { return T(10); }

	static T eval_scalar(const T x, const T y) { return x - y; }

	static simd_pack<T, sse_kind> eval_vector(const simd_pack<T, sse_kind> x, const simd_pack<T, sse_kind> y)
	{
		return x - y;
	}
};


template<typename T>
struct mul_ts
{
	static const char *name() { return "mul"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T lb_y() { return T(-10); }
	static T ub_y() { return T(10); }

	static T eval_scalar(const T x, const T y) { return x * y; }

	static simd_pack<T, sse_kind> eval_vector(const simd_pack<T, sse_kind> x, const simd_pack<T, sse_kind> y)
	{
		return x * y;
	}
};


template<typename T>
struct div_ts
{
	static const char *name() { return "div"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T lb_y() { return T(1); }
	static T ub_y() { return T(10); }

	static T eval_scalar(const T x, const T y) { return x / y; }

	static simd_pack<T, sse_kind> eval_vector(const simd_pack<T, sse_kind> x, const simd_pack<T, sse_kind> y)
	{
		return x / y;
	}
};


template<typename T>
struct neg_ts
{
	static const char *name() { return "neg"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return -x; }

	static simd_pack<T, sse_kind> eval_vector(const simd_pack<T, sse_kind> x)
	{
		return - x;
	}
};


template<typename T>
struct abs_ts
{
	static const char *name() { return "abs"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return std::fabs(x); }

	static simd_pack<T, sse_kind> eval_vector(const simd_pack<T, sse_kind> x)
	{
		return abs(x);
	}
};


template<typename T>
struct min_ts
{
	static const char *name() { return "min"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T lb_y() { return T(1); }
	static T ub_y() { return T(10); }

	static T eval_scalar(const T x, const T y) { return x < y ? x : y; }

	static simd_pack<T, sse_kind> eval_vector(const simd_pack<T, sse_kind> x, const simd_pack<T, sse_kind> y)
	{
		return vmin(x, y);
	}
};


template<typename T>
struct max_ts
{
	static const char *name() { return "max"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T lb_y() { return T(1); }
	static T ub_y() { return T(10); }

	static T eval_scalar(const T x, const T y) { return x > y ? x : y; }

	static simd_pack<T, sse_kind> eval_vector(const simd_pack<T, sse_kind> x, const simd_pack<T, sse_kind> y)
	{
		return vmax(x, y);
	}
};


template<typename T>
struct sqr_ts
{
	static const char *name() { return "sqr"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return x * x; }

	static simd_pack<T, sse_kind> eval_vector(const simd_pack<T, sse_kind> x)
	{
		return sqr(x);
	}
};


template<typename T>
struct sqrt_ts
{
	static const char *name() { return "sqrt"; }

	static T lb_x() { return T(0.1); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return std::sqrt(x); }

	static simd_pack<T, sse_kind> eval_vector(const simd_pack<T, sse_kind> x)
	{
		return sqrt(x);
	}
};


template<typename T>
struct rcp_ts
{
	static const char *name() { return "rcp"; }

	static T lb_x() { return T(1); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return T(1) / x; }

	static simd_pack<T, sse_kind> eval_vector(const simd_pack<T, sse_kind> x)
	{
		return rcp(x);
	}
};


template<typename T>
struct rsqrt_ts
{
	static const char *name() { return "rsqrt"; }

	static T lb_x() { return T(1); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return T(1) / std::sqrt(x); }

	static simd_pack<T, sse_kind> eval_vector(const simd_pack<T, sse_kind> x)
	{
		return rsqrt(x);
	}
};


template<typename T>
struct rcp_a_ts
{
	static const char *name() { return "rcp(a)"; }

	static T lb_x() { return T(1); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return T(1) / x; }

	static simd_pack<T, sse_kind> eval_vector(const simd_pack<T, sse_kind> x)
	{
		return approx_rcp(x.impl);
	}
};


template<typename T>
struct rsqrt_a_ts
{
	static const char *name() { return "rsqrt(a)"; }

	static T lb_x() { return T(1); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return T(1) / std::sqrt(x); }

	static simd_pack<T, sse_kind> eval_vector(const simd_pack<T, sse_kind> x)
	{
		return approx_rsqrt(x.impl);
	}
};


template<typename T>
struct cube_ts
{
	static const char *name() { return "cube"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return x * x * x; }

	static simd_pack<T, sse_kind> eval_vector(const simd_pack<T, sse_kind> x)
	{
		return cube(x);
	}
};


template<typename T>
struct floor_ts
{
	static const char *name() { return "floor"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return std::floor(x); }

	static simd_pack<T, sse_kind> eval_vector(const simd_pack<T, sse_kind> x)
	{
		return floor(x);
	}
};

template<typename T>
struct ceil_ts
{
	static const char *name() { return "ceil"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return std::ceil(x); }

	static simd_pack<T, sse_kind> eval_vector(const simd_pack<T, sse_kind> x)
	{
		return ceil(x);
	}
};

template<typename T>
struct floor2_ts
{
	static const char *name() { return "floor(2)"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return std::floor(x); }

	static simd_pack<T, sse_kind> eval_vector(const simd_pack<T, sse_kind> x)
	{
		return floor_sse2(x.impl);
	}
};

template<typename T>
struct ceil2_ts
{
	static const char *name() { return "ceil(2)"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return std::ceil(x); }

	static simd_pack<T, sse_kind> eval_vector(const simd_pack<T, sse_kind> x)
	{
		return ceil_sse2(x.impl);
	}
};


template<typename T>
bool test_all()
{
	bool passed = true;

	if (!test_accuracy_b<T, add_ts>()) passed = false;
	if (!test_accuracy_b<T, sub_ts>()) passed = false;
	if (!test_accuracy_b<T, mul_ts>()) passed = false;
	if (!test_accuracy_b<T, div_ts>()) passed = false;

	if (!test_accuracy_u<T, neg_ts>()) passed = false;
	if (!test_accuracy_u<T, abs_ts>()) passed = false;
	if (!test_accuracy_b<T, min_ts>()) passed = false;
	if (!test_accuracy_b<T, max_ts>()) passed = false;

	if (!test_accuracy_u<T, sqr_ts>()) passed = false;
	if (!test_accuracy_u<T, sqrt_ts>()) passed = false;
	if (!test_accuracy_u<T, rcp_ts>()) passed = false;
	if (!test_accuracy_u<T, rsqrt_ts>()) passed = false;
	if (!test_accuracy_u<T, cube_ts>()) passed = false;

	if (!test_accuracy_u<T, floor_ts>()) passed = false;
	if (!test_accuracy_u<T, ceil_ts>()) passed = false;
	if (!test_accuracy_u<T, floor2_ts>()) passed = false;
	if (!test_accuracy_u<T, ceil2_ts>()) passed = false;

	return passed;
}

#ifdef _MSC_VER
#pragma warning(disable: 4100)
#endif


int main(int argc, char *argv[])
{
	bool passed = true;

	std::printf("Tests on f32\n");
	std::printf("================================\n");
	if (!test_all<f32>()) passed = false;

	if (!test_accuracy_u<f32, rcp_a_ts>(1.e-3f)) passed = false;
	if (!test_accuracy_u<f32, rsqrt_a_ts>(1.e-3f)) passed = false;

	std::printf("\n");

	std::printf("Tests on f64\n");
	std::printf("================================\n");
	if (!test_all<f64>()) passed = false;

	std::printf("\n");

	return passed ? 0 : -1;
}


