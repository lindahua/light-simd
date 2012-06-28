/**
 * @file test_sse_mm.cpp
 *
 * Unit testing for SSE matrix-multiplication
 *
 * @author Dahua Lin
 */


#include "test_aux.h"
#include "linalg_ref.h"

using namespace lsimd;
using namespace ltest;

#ifdef _MSC_VER
#pragma warning(disable : 4324 4996)
#endif

const int MaxArrLen = 16;

LSIMD_ALIGN(32) f32 arr_af[MaxArrLen];
LSIMD_ALIGN(32) f32 arr_bf[MaxArrLen];
LSIMD_ALIGN(32) f32 arr_crf[MaxArrLen];
LSIMD_ALIGN(32) f32 arr_c0f[MaxArrLen];

LSIMD_ALIGN(32) f64 arr_ad[MaxArrLen];
LSIMD_ALIGN(32) f64 arr_bd[MaxArrLen];
LSIMD_ALIGN(32) f64 arr_crd[MaxArrLen];
LSIMD_ALIGN(32) f64 arr_c0d[MaxArrLen];

template<typename T> struct storage_s;

template<> struct storage_s<f32>
{
	static f32 *arr_a() { return arr_af; }
	static f32 *arr_b() { return arr_bf; }
	static f32 *arr_cr() { return arr_crf; }
	static f32 *arr_c0() { return arr_c0f; }
};

template<> struct storage_s<f64>
{
	static f64 *arr_a() { return arr_ad; }
	static f64 *arr_b() { return arr_bd; }
	static f64 *arr_cr() { return arr_crd; }
	static f64 *arr_c0() { return arr_c0d; }
};


template<typename T, int M, int K, int N>
class matmul_tests : public test_case
{
	char m_name[128];

	T *arr_a;
	T *arr_b;
	T *arr_cr;
	T *arr_c0;
	
public:
	matmul_tests()
	{
		std::sprintf(m_name, "mm (%d x %d) * (%d x %d)", M, K, K, N);

		arr_a = storage_s<T>::arr_a();
		arr_b = storage_s<T>::arr_b();
		arr_cr = storage_s<T>::arr_cr();
		arr_c0 = storage_s<T>::arr_c0();
	}

	const char *name() const
	{
		return m_name;
	}

	void run()
	{
		// compute ground-truth

		simple_mat<T, M, K> a0(arr_a);
		simple_mat<T, K, N> b0(arr_b);
		simple_mat<T, M, N> c0(arr_c0);

		for (int i = 0; i < M * K; ++i) a0[i] = T(i + 1);
		for (int i = 0; i < K * N; ++i) b0[i] = T(i + 2);
		for (int i = 0; i < M * N; ++i) c0[i] = T(-1);

		ref_mm(a0, b0, c0);

		// use SIMD

		simd_mat<T, M, K, sse_kind> a( arr_a, aligned_t() );
		simd_mat<T, K, N, sse_kind> b( arr_b, aligned_t() );
		simd_mat<T, M, N, sse_kind> c = a * b;

		c.store( arr_cr, aligned_t() );

		if (!test_vector_equal(M * N, arr_c0, arr_cr))
		{
			std::printf("\n");
			simple_mat<T, M, N> cr(arr_cr);

			a0.print("%5g ");
			std::printf("*\n");
			b0.print("%5g ");
			std::printf("==>\n");
			c0.print("%5g ");
			std::printf("actual result = \n");
			cr.print("%5g ");
			std::printf("\n");
		}

		ASSERT_VEC_EQ( M * N, arr_c0, arr_cr );
	}
};


template<typename T>
void add_cases_to_matmul_tpack(test_pack* tp)
{
	tp->add( new matmul_tests<T, 2, 2, 2>() );
	tp->add( new matmul_tests<T, 2, 2, 3>() );
	tp->add( new matmul_tests<T, 2, 2, 4>() );

	tp->add( new matmul_tests<T, 2, 3, 2>() );
	tp->add( new matmul_tests<T, 2, 3, 3>() );
	tp->add( new matmul_tests<T, 2, 3, 4>() );

	tp->add( new matmul_tests<T, 2, 4, 2>() );
	tp->add( new matmul_tests<T, 2, 4, 3>() );
	tp->add( new matmul_tests<T, 2, 4, 4>() );

	tp->add( new matmul_tests<T, 3, 2, 2>() );
	tp->add( new matmul_tests<T, 3, 2, 3>() );
	tp->add( new matmul_tests<T, 3, 2, 4>() );

	tp->add( new matmul_tests<T, 3, 3, 2>() );
	tp->add( new matmul_tests<T, 3, 3, 3>() );
	tp->add( new matmul_tests<T, 3, 3, 4>() );

	tp->add( new matmul_tests<T, 3, 4, 2>() );
	tp->add( new matmul_tests<T, 3, 4, 3>() );
	tp->add( new matmul_tests<T, 3, 4, 4>() );

	tp->add( new matmul_tests<T, 4, 2, 2>() );
	tp->add( new matmul_tests<T, 4, 2, 3>() );
	tp->add( new matmul_tests<T, 4, 2, 4>() );

	tp->add( new matmul_tests<T, 4, 3, 2>() );
	tp->add( new matmul_tests<T, 4, 3, 3>() );
	tp->add( new matmul_tests<T, 4, 3, 4>() );

	tp->add( new matmul_tests<T, 4, 4, 2>() );
	tp->add( new matmul_tests<T, 4, 4, 3>() );
	tp->add( new matmul_tests<T, 4, 4, 4>() );
}


test_pack* matmul_tpack_f32()
{
	test_pack *tp = new test_pack( "matmul_f32" );
	add_cases_to_matmul_tpack<f32>(tp);
	return tp;
}

test_pack* matmul_tpack_f64()
{
	test_pack *tp = new test_pack( "matmul_f64" );
	add_cases_to_matmul_tpack<f64>(tp);
	return tp;
}


void lsimd::add_test_packs()
{
	lsimd_main_suite.add( matmul_tpack_f32() );
	lsimd_main_suite.add( matmul_tpack_f64() );
}







