/*
 * @file test_sse_sol.cpp
 *
 * Unit testing of matrix inverse and equaltion solving
 *
 * @author Dahua Lin
 */


#include "test_aux.h"
#include "linalg_ref.h"

using namespace lsimd;
using namespace ltest;

template<typename T>
void special_fill_mat(int n, T *x)
{
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			T b = T(4 - j);

			T v = b;
			for (int k = 0; k < i; ++k) v *= b;

			x[i + j * n] = v;
		}
	}
}

template<typename T, int N>
T special_det()
{
	switch (N)
	{
	case 2: return T(-12);
	case 3: return T(-48);
	case 4: return T(288);
	}

	return T(0);
}


GCASE1( det )
{
	LSIMD_ALIGN_SSE T src[N * N];
	special_fill_mat(N, src);

	simd_mat<T, N, N, sse_kind> a(src, aligned_t());

	simple_mat<T, N, N> a0(src);

	T v0 = special_det<T, N>();

	// std::printf("det(a) --> %g\n", det(a));
	ASSERT_EQ( det(a), v0 );
}


GCASE1( inv )
{
	LSIMD_ALIGN_SSE T av[N * N];
	LSIMD_ALIGN_SSE T bv[N * N];
	special_fill_mat(N, av);

	simd_mat<T, N, N, sse_kind> a(av, aligned_t());
	simd_mat<T, N, N, sse_kind> inv_a = inv(a);
	inv_a.store(bv, aligned_t());

	T E[N * N];
	T E0[N * N];

	fill_const(N * N, E, T(-1));

	fill_const(N * N, E0, T(0));
	for (int i = 0; i < N; ++i) E0[i + i * N] = T(1);

	simple_mat<T,N,N> am(av);
	simple_mat<T,N,N> bm(bv);
	simple_mat<T,N,N> cm(E);
	ref_mm(am, bm, cm);

	T tol = sizeof(T) == 4 ? T(1.0e-4) : T(1.0e-12);
	ASSERT_VEC_APPROX(N*N, E, E0, tol);

	simd_mat<T, N, N, sse_kind> inv_a2;
	T detv = inv_and_det(a, inv_a2);

	T detv0 = special_det<T, N>();

	ASSERT_EQ( detv, detv0 );

	inv_a2.store(bv, aligned_t());
	ref_mm(am, bm, cm);
	ASSERT_VEC_APPROX(N*N, E, E0, tol);
}


GCASE1( solve )
{
	T tol = sizeof(T) == 4 ? T(5.0e-5) : T(1.0e-12);

	LSIMD_ALIGN_SSE T av[N * N];
	LSIMD_ALIGN_SSE T bv[N];
	LSIMD_ALIGN_SSE T yv[N];

	special_fill_mat(N, av);
	for (int i = 0; i < N; ++i) bv[i] = T(i+1);

	simd_mat<T, N, N, sse_kind> A(av, aligned_t());

	simd_vec<T, N, sse_kind> b(bv, aligned_t());
	simd_vec<T, N, sse_kind> x = solve(A, b);

	fill_const(N, yv, T(-1));
	(A * x).store(yv, aligned_t());

	ASSERT_VEC_APPROX(N, yv, bv, tol);
}


GCASE2( solve_mat )
{
	T tol = sizeof(T) == 4 ? T(5.0e-4) : T(1.0e-12);

	LSIMD_ALIGN_SSE T av[M * M];
	LSIMD_ALIGN_SSE T bv[M * N];
	LSIMD_ALIGN_SSE T yv[M * N];

	special_fill_mat(M, av);
	for (int i = 0; i < M * N; ++i) bv[i] = T(i+1);

	simd_mat<T, M, M, sse_kind> A(av, aligned_t());

	simd_mat<T, M, N, sse_kind> B(bv, aligned_t());
	simd_mat<T, M, N, sse_kind> X = solve(A, B);

	fill_const(M * N, yv, T(-1));
	(A * X).store(yv, aligned_t());

	ASSERT_VEC_APPROX(M * N, yv, bv, tol);
}



test_pack* det_tpack()
{
	test_pack *tp = new test_pack( "det" );

	tp->add( new det_tests<f32, 2>() );
	tp->add( new det_tests<f64, 2>() );

	tp->add( new det_tests<f32, 3>() );
	tp->add( new det_tests<f64, 3>() );

	tp->add( new det_tests<f32, 4>() );
	tp->add( new det_tests<f64, 4>() );

	return tp;
}

test_pack* inv_tpack()
{
	test_pack *tp = new test_pack( "inv" );

	tp->add( new inv_tests<f32, 2>() );
	tp->add( new inv_tests<f64, 2>() );

	tp->add( new inv_tests<f32, 3>() );
	tp->add( new inv_tests<f64, 3>() );

	tp->add( new inv_tests<f32, 4>() );
	tp->add( new inv_tests<f64, 4>() );

	return tp;
}

test_pack* solve_tpack()
{
	test_pack *tp = new test_pack( "solve" );

	tp->add( new solve_tests<f32, 2>() );
	tp->add( new solve_tests<f64, 2>() );

	tp->add( new solve_tests<f32, 3>() );
	tp->add( new solve_tests<f64, 3>() );

	tp->add( new solve_tests<f32, 4>() );
	tp->add( new solve_tests<f64, 4>() );

	return tp;
}


test_pack* solve_mat_tpack()
{
	test_pack *tp = new test_pack( "solve_mat" );

	tp->add( new solve_mat_tests<f32, 2, 2>() );
	tp->add( new solve_mat_tests<f32, 2, 3>() );
	tp->add( new solve_mat_tests<f32, 2, 4>() );

	tp->add( new solve_mat_tests<f64, 2, 2>() );
	tp->add( new solve_mat_tests<f64, 2, 3>() );
	tp->add( new solve_mat_tests<f64, 2, 4>() );

	tp->add( new solve_mat_tests<f32, 3, 2>() );
	tp->add( new solve_mat_tests<f32, 3, 3>() );
	tp->add( new solve_mat_tests<f32, 3, 4>() );

	tp->add( new solve_mat_tests<f64, 3, 2>() );
	tp->add( new solve_mat_tests<f64, 3, 3>() );
	tp->add( new solve_mat_tests<f64, 3, 4>() );

	tp->add( new solve_mat_tests<f32, 4, 2>() );
	tp->add( new solve_mat_tests<f32, 4, 3>() );
	tp->add( new solve_mat_tests<f32, 4, 4>() );

	tp->add( new solve_mat_tests<f64, 4, 2>() );
	tp->add( new solve_mat_tests<f64, 4, 3>() );
	tp->add( new solve_mat_tests<f64, 4, 4>() );

	return tp;
}


void lsimd::add_test_packs()
{
	lsimd_main_suite.add( det_tpack() );
	lsimd_main_suite.add( inv_tpack() );
	lsimd_main_suite.add( solve_tpack() );
	lsimd_main_suite.add( solve_mat_tpack() );
}








