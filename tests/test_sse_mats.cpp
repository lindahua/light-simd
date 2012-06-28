/**
 * @file test_sse_mats.cpp
 *
 * Test the correctness of sse_mat classes
 *
 * @author Dahua Lin
 */


#include "test_aux.h"

using namespace lsimd;
using namespace ltest;

// explicit instantiation for thorough syntax check
/*
template class lsimd::sse_mat<f32, 2, 2>;
template class lsimd::sse_mat<f32, 2, 3>;
template class lsimd::sse_mat<f32, 2, 4>;
template class lsimd::sse_mat<f32, 3, 2>;
template class lsimd::sse_mat<f32, 3, 3>;
template class lsimd::sse_mat<f32, 3, 4>;
template class lsimd::sse_mat<f32, 4, 2>;
template class lsimd::sse_mat<f32, 4, 3>;
template class lsimd::sse_mat<f32, 4, 4>;

template class lsimd::sse_mat<f64, 2, 2>;
template class lsimd::sse_mat<f64, 2, 3>;
template class lsimd::sse_mat<f64, 2, 4>;
template class lsimd::sse_mat<f64, 3, 2>;
template class lsimd::sse_mat<f64, 3, 3>;
template class lsimd::sse_mat<f64, 3, 4>;
template class lsimd::sse_mat<f64, 4, 2>;
template class lsimd::sse_mat<f64, 4, 3>;
template class lsimd::sse_mat<f64, 4, 4>;


template struct lsimd::simd_mat<f32, 2, 2, sse_kind>;
template struct lsimd::simd_mat<f32, 2, 3, sse_kind>;
template struct lsimd::simd_mat<f32, 2, 4, sse_kind>;
template struct lsimd::simd_mat<f32, 3, 2, sse_kind>;
template struct lsimd::simd_mat<f32, 3, 3, sse_kind>;
template struct lsimd::simd_mat<f32, 3, 4, sse_kind>;
template struct lsimd::simd_mat<f32, 4, 2, sse_kind>;
template struct lsimd::simd_mat<f32, 4, 3, sse_kind>;
template struct lsimd::simd_mat<f32, 4, 4, sse_kind>;

template struct lsimd::simd_mat<f64, 2, 2, sse_kind>;
template struct lsimd::simd_mat<f64, 2, 3, sse_kind>;
template struct lsimd::simd_mat<f64, 2, 4, sse_kind>;
template struct lsimd::simd_mat<f64, 3, 2, sse_kind>;
template struct lsimd::simd_mat<f64, 3, 3, sse_kind>;
template struct lsimd::simd_mat<f64, 3, 4, sse_kind>;
template struct lsimd::simd_mat<f64, 4, 2, sse_kind>;
template struct lsimd::simd_mat<f64, 4, 3, sse_kind>;
template struct lsimd::simd_mat<f64, 4, 4, sse_kind>;
*/

const int MaxArrLen = 36;
const int LDa = 8;
const int LDu = 5;

GCASE2( zero )
{
	T r[MaxArrLen];
	fill_const(MaxArrLen, r, T(0));

	simd_mat<T, M, N, sse_kind> a = zero_t();

	ASSERT_SIMD_EQ( a, r );
}


GCASE2( load )
{
	LSIMD_ALIGN_SSE T src[MaxArrLen];
	for (int i = 0; i < MaxArrLen; ++i) src[i] = T(i+1);

	simd_mat<T, M, N, sse_kind> aa(src, aligned_t());
	ASSERT_SIMD_EQ( aa, src );

	simd_mat<T, M, N, sse_kind> au(src + 1, unaligned_t());
	ASSERT_SIMD_EQ( au, src + 1 );

	T br[M * N];

	for (int j = 0; j < N; ++j)
		for (int i = 0; i < M; ++i) br[i + j * M] = src[i + j * LDa];

	simd_mat<T, M, N, sse_kind> ba(src, LDa, aligned_t());
	ASSERT_SIMD_EQ( ba, br );

	for (int j = 0; j < N; ++j)
		for (int i = 0; i < M; ++i) br[i + j * M] = src[1 + i + j * LDu];

	simd_mat<T, M, N, sse_kind> bu(src+1, LDu, unaligned_t());
	ASSERT_SIMD_EQ( bu, br );
}


GCASE2( store )
{
	LSIMD_ALIGN_SSE T src[MaxArrLen];
	for (int i = 0; i < MaxArrLen; ++i) src[i] = T(i+1);

	LSIMD_ALIGN_SSE T da[MaxArrLen];
	T dd[MaxArrLen];
	T r[MaxArrLen];

	simd_mat<T, M, N, sse_kind> a(src, aligned_t());
	ASSERT_SIMD_EQ(a, src);

	// store continuous align

	for (int i = 0; i < MaxArrLen; ++i) da[i] = T(-1);
	for (int i = 0; i < MaxArrLen; ++i) r[i] = T(-1);
	for (int i = 0; i < M * N; ++i) r[i] = src[i];

	a.store(da, aligned_t());

	ASSERT_VEC_EQ(MaxArrLen, da, r);

	// store continuous non-align

	for (int i = 0; i < MaxArrLen; ++i) dd[i] = T(-1);
	for (int i = 0; i < MaxArrLen; ++i) r[i] = T(-1);
	for (int i = 0; i < M * N; ++i) r[i+1] = src[i];

	a.store(dd + 1, unaligned_t());

	ASSERT_VEC_EQ(MaxArrLen, dd, r);

	// store non-continuous align

	for (int i = 0; i < MaxArrLen; ++i) da[i] = T(-1);
	for (int i = 0; i < MaxArrLen; ++i) r[i] = T(-1);
	for (int j = 0; j < N; ++j)
		for (int i = 0; i < M; ++i) r[i + j * LDa] = src[i + j * M];

	a.store(da, LDa, aligned_t());

	ASSERT_VEC_EQ(MaxArrLen, da, r);

	// store non-continuous non-align

	for (int i = 0; i < MaxArrLen; ++i) dd[i] = T(-1);
	for (int i = 0; i < MaxArrLen; ++i) r[i] = T(-1);
	for (int j = 0; j < N; ++j)
		for (int i = 0; i < M; ++i) r[1 + i + j * LDu] = src[i + j * M];

	a.store(dd + 1, LDu, unaligned_t());

	ASSERT_VEC_EQ(MaxArrLen, dd, r);
}


GCASE2( load_trans )
{
	LSIMD_ALIGN_SSE T src[MaxArrLen];
	for (int i = 0; i < MaxArrLen; ++i) src[i] = T(i+1);
	T r[M * N];

	simd_mat<T, M, N, sse_kind> a;

	a.load_trans(src, aligned_t());

	for (int j = 0; j  < N; ++j)
		for (int i = 0; i < M; ++i) r[i + j * M] = src[i * N + j];

	ASSERT_SIMD_EQ(a, r);

	a.load_trans(src + 1, unaligned_t());

	for (int j = 0; j  < N; ++j)
		for (int i = 0; i < M; ++i) r[i + j * M] = src[1 + i * N + j];

	ASSERT_SIMD_EQ(a, r);

	a.load_trans(src, LDa, aligned_t());

	for (int j = 0; j  < N; ++j)
		for (int i = 0; i < M; ++i) r[i + j * M] = src[i * LDa + j];

	ASSERT_SIMD_EQ(a, r);

	a.load_trans(src + 1, LDu, unaligned_t());

	for (int j = 0; j  < N; ++j)
		for (int i = 0; i < M; ++i) r[i + j * M] = src[1 + i * LDu + j];

	ASSERT_SIMD_EQ(a, r);
}


GCASE2( arith )
{
	LSIMD_ALIGN_SSE T sa[MaxArrLen];
	for (int i = 0; i < MaxArrLen; ++i) sa[i] = T(i+1);

	LSIMD_ALIGN_SSE T sb[MaxArrLen];
	for (int i = 0; i < MaxArrLen; ++i) sb[i] = T(MaxArrLen - 2 * i);

	T r[M * N];

	simd_mat<T, M, N, sse_kind> a(sa, aligned_t());
	simd_mat<T, M, N, sse_kind> a2;
	simd_mat<T, M, N, sse_kind> b(sb, aligned_t());

	for (int i = 0; i < M * N; ++i) r[i] = sa[i] + sb[i];
	ASSERT_SIMD_EQ(a + b, r);

	a2 = a;
	a2 += b;
	ASSERT_SIMD_EQ(a2, r);

	for (int i = 0; i < M * N; ++i) r[i] = sa[i] - sb[i];
	ASSERT_SIMD_EQ(a - b, r);

	a2 = a;
	a2 -= b;
	ASSERT_SIMD_EQ(a2, r);

	for (int i = 0; i < M * N; ++i) r[i] = sa[i] * sb[i];
	ASSERT_SIMD_EQ(a % b, r);

	a2 = a;
	a2 %= b;
	ASSERT_SIMD_EQ(a2, r);
}


GCASE2( scale )
{
	LSIMD_ALIGN_SSE T sa[MaxArrLen];
	for (int i = 0; i < MaxArrLen; ++i) sa[i] = T(i+1);

	T b = T(2.5);
	simd_pack<T, sse_kind> bp(b);

	T r[M * N];

	simd_mat<T, M, N, sse_kind> a(sa, aligned_t());

	for (int i = 0; i < M * N; ++i) r[i] = sa[i] * b;
	ASSERT_SIMD_EQ(a * bp, r);

	a *= bp;
	ASSERT_SIMD_EQ(a, r);
}



GCASE2( mtimes )
{
	LSIMD_ALIGN_SSE T sa[MaxArrLen];
	for (int i = 0; i < MaxArrLen; ++i) sa[i] = T(i+1);

	LSIMD_ALIGN_SSE T sx[N];
	for (int j = 0; j < N; ++j) sx[j] = T(j+1);

	T r[M];
	for (int i = 0; i < M; ++i)
	{
		T s(0);
		for (int j = 0; j < N; ++j) s += sa[i + j * M] * sx[j];
		r[i] = s;
	}

	simd_mat<T, M, N, sse_kind> a(sa, aligned_t());
	simd_vec<T, N, sse_kind> x(sx, aligned_t());

	ASSERT_SIMD_EQ( a * x, r );
}


GCASE2( trace )
{
	LSIMD_ALIGN_SSE T sa[MaxArrLen];
	for (int i = 0; i < MaxArrLen; ++i) sa[i] = T(i+1);

	T t(0);
	int d = (M < N ? M : N);

	for (int i = 0; i < d; ++i)
	{
		t += sa[i + i * M];
	}

	simd_mat<T, M, N, sse_kind> a(sa, aligned_t());
	ASSERT_EQ( a.trace(), t );
}



template<template<typename T, int M, int N> class H>
test_pack* make_tpack( const char *name )
{
	test_pack *tp = new test_pack( name );

	// for f32

	tp->add( new H<f32, 2, 2>() );
	tp->add( new H<f32, 2, 3>() );
	tp->add( new H<f32, 2, 4>() );

	tp->add( new H<f32, 3, 2>() );
	tp->add( new H<f32, 3, 3>() );
	tp->add( new H<f32, 3, 4>() );

	tp->add( new H<f32, 4, 2>() );
	tp->add( new H<f32, 4, 3>() );
	tp->add( new H<f32, 4, 4>() );

	// for f64

	tp->add( new H<f64, 2, 2>() );
	tp->add( new H<f64, 2, 3>() );
	tp->add( new H<f64, 2, 4>() );

	tp->add( new H<f64, 3, 2>() );
	tp->add( new H<f64, 3, 3>() );
	tp->add( new H<f64, 3, 4>() );

	tp->add( new H<f64, 4, 2>() );
	tp->add( new H<f64, 4, 3>() );
	tp->add( new H<f64, 4, 4>() );

	return tp;
}


#define ADD_TEST( name ) lsimd_main_suite.add( make_tpack<name##_tests>( #name ) )

void lsimd::add_test_packs()
{
	ADD_TEST( zero );
	ADD_TEST( load );
	ADD_TEST( store );
	ADD_TEST( load_trans );
	ADD_TEST( arith );
	ADD_TEST( scale );
	ADD_TEST( mtimes );
	ADD_TEST( trace );
}




