/**
 * @file test_sse_vecs.cpp
 *
 * Testing the correctness of sse_vec classes
 *
 * @author Dahua Lin
 */


#include "test_aux.h"

using namespace lsimd;
using namespace ltest;

// explicit instantiation for thorough syntax check

template struct lsimd::simd_vec<f32, 1, sse_kind>;
template struct lsimd::simd_vec<f32, 2, sse_kind>;
template struct lsimd::simd_vec<f32, 3, sse_kind>;
template struct lsimd::simd_vec<f32, 4, sse_kind>;

template struct lsimd::simd_vec<f64, 1, sse_kind>;
template struct lsimd::simd_vec<f64, 2, sse_kind>;
template struct lsimd::simd_vec<f64, 3, sse_kind>;
template struct lsimd::simd_vec<f64, 4, sse_kind>;



const int MaxVLen = 4;

// Test cases


// zero

GCASE1( zero )
{
	T r[N];
	fill_const(N, r, T(0));

	simd_vec<T, N, sse_kind> v = zero_t();
	ASSERT_SIMD_EQ(v, r);
}

// set

template<typename T, int N> class set_tests;

SCASE1( set, 1 )
{
	T a[1] = { T(1.1) };

	sse_vec<T, 1> v(a[0]);
	ASSERT_TRUE( v.test_equal(a) );
}

SCASE1( set, 2 )
{
	T a[2] = { T(1.1), T(2.2) };

	sse_vec<T, 2> v(a[0], a[1]);
	ASSERT_TRUE( v.test_equal(a) );
}

SCASE1( set, 3 )
{
	T a[3] = { T(1.1), T(2.2), T(3.3) };

	sse_vec<T, 3> v(a[0], a[1], a[2]);
	ASSERT_TRUE( v.test_equal(a) );
}

SCASE1( set, 4 )
{
	T a[4] = { T(1.1), T(2.2), T(3.3), T(4.4) };

	sse_vec<T, 4> v(a[0], a[1], a[2], a[3]);
	ASSERT_TRUE( v.test_equal(a) );
}


// load

GCASE1( load )
{
	LSIMD_ALIGN_SSE T src[MaxVLen + 1] = { T(1.1), T(2.2), T(3.3), T(4.4), T(5.5) };

	simd_vec<T, N, sse_kind> v1(src, aligned_t());
	ASSERT_SIMD_EQ( v1, src );

	simd_vec<T, N, sse_kind> v2(src + 1, unaligned_t());
	ASSERT_SIMD_EQ( v2, src + 1 );

	simd_vec<T, N, sse_kind> v3;
	v3.load( src, aligned_t() );
	ASSERT_SIMD_EQ( v3, src );

	simd_vec<T, N, sse_kind> v4;
	v4.load( src + 1, unaligned_t() );
	ASSERT_SIMD_EQ( v4, src + 1 );
}


// store

GCASE1( store )
{
	LSIMD_ALIGN_SSE T src[MaxVLen] = { T(1.1), T(2.2), T(3.3), T(4.4) };
	LSIMD_ALIGN_SSE T dst[MaxVLen + 1];

	simd_vec<T, N, sse_kind> v(src, aligned_t());

	T r1[5];
	fill_const(5, r1, T(-1));
	for (int i = 0; i < N; ++i) r1[i] = src[i];

	fill_const(5, dst, T(-1));
	v.store(dst, aligned_t());

	ASSERT_VEC_EQ(5, dst, r1);

	T r2[5];
	fill_const(5, r2, T(-1));
	for (int i = 0; i < N; ++i) r2[i+1] = src[i];

	fill_const(5, dst, T(-1));
	v.store(dst + 1, unaligned_t());

	ASSERT_VEC_EQ(5, dst, r2);
}


// set

template<typename T, int N> class bsxp_tests;

SCASE1( bsxp, 1 )
{
	T a[1] = { T(1.1) };
	const int w = (int)simd<T, sse_kind>::pack_width;

	sse_vec<T, 1> v(a[0]);

	T r[w];

	fill_const(w, r, a[0]);
	sse_pack<T> p = v.template bsx_pk<0>();
	ASSERT_TRUE( p.test_equal(r) );
}


SCASE1( bsxp, 2 )
{
	T a[2] = { T(1.1), T(2.2) };
	const int w = (int)simd<T, sse_kind>::pack_width;

	sse_vec<T, 2> v(a[0], a[1]);

	T r[w];

	fill_const(w, r, a[0]);
	sse_pack<T> p0 = v.template bsx_pk<0>();
	ASSERT_TRUE( p0.test_equal(r) );

	fill_const(w, r, a[1]);
	sse_pack<T> p1 = v.template bsx_pk<1>();
	ASSERT_TRUE( p1.test_equal(r) );
}

SCASE1( bsxp, 3 )
{
	T a[3] = { T(1.1), T(2.2), T(3.3) };
	const int w = (int)simd<T, sse_kind>::pack_width;

	sse_vec<T, 3> v(a[0], a[1], a[2]);

	T r[w];

	fill_const(w, r, a[0]);
	sse_pack<T> p0 = v.template bsx_pk<0>();
	ASSERT_TRUE( p0.test_equal(r) );

	fill_const(w, r, a[1]);
	sse_pack<T> p1 = v.template bsx_pk<1>();
	ASSERT_TRUE( p1.test_equal(r) );

	fill_const(w, r, a[2]);
	sse_pack<T> p2 = v.template bsx_pk<2>();
	ASSERT_TRUE( p2.test_equal(r) );
}

SCASE1( bsxp, 4 )
{
	T a[4] = { T(1.1), T(2.2), T(3.3), T(4.4) };
	const int w = (int)simd<T, sse_kind>::pack_width;

	sse_vec<T, 4> v(a[0], a[1], a[2], a[3]);

	T r[w];

	fill_const(w, r, a[0]);
	sse_pack<T> p0 = v.template bsx_pk<0>();
	ASSERT_TRUE( p0.test_equal(r) );

	fill_const(w, r, a[1]);
	sse_pack<T> p1 = v.template bsx_pk<1>();
	ASSERT_TRUE( p1.test_equal(r) );

	fill_const(w, r, a[2]);
	sse_pack<T> p2 = v.template bsx_pk<2>();
	ASSERT_TRUE( p2.test_equal(r) );

	fill_const(w, r, a[3]);
	sse_pack<T> p3 = v.template bsx_pk<3>();
	ASSERT_TRUE( p3.test_equal(r) );
}



// arithmetic

GCASE1( add )
{
	LSIMD_ALIGN_SSE T a[MaxVLen] = { T(1), T(2), T(3), T(4) };
	LSIMD_ALIGN_SSE T b[MaxVLen] = { T(3), T(7), T(2), T(6) };

	T r[N];
	for (int i = 0; i < N; ++i) r[i] = a[i] + b[i];

	simd_vec<T, N, sse_kind> va(a, aligned_t());
	simd_vec<T, N, sse_kind> vb(b, aligned_t());

	ASSERT_SIMD_EQ( va + vb, r);

	va += vb;
	ASSERT_SIMD_EQ( va, r);
}


GCASE1( sub )
{
	LSIMD_ALIGN_SSE T a[MaxVLen] = { T(1), T(2), T(3), T(4) };
	LSIMD_ALIGN_SSE T b[MaxVLen] = { T(3), T(7), T(2), T(6) };

	T r[N];
	for (int i = 0; i < N; ++i) r[i] = a[i] - b[i];

	simd_vec<T, N, sse_kind> va(a, aligned_t());
	simd_vec<T, N, sse_kind> vb(b, aligned_t());

	ASSERT_SIMD_EQ( va - vb, r);

	va -= vb;
	ASSERT_SIMD_EQ( va, r);
}


GCASE1( mul )
{
	LSIMD_ALIGN_SSE T a[MaxVLen] = { T(1), T(2), T(3), T(4) };
	LSIMD_ALIGN_SSE T b[MaxVLen] = { T(3), T(7), T(2), T(6) };

	T r[N];
	for (int i = 0; i < N; ++i) r[i] = a[i] * b[i];

	simd_vec<T, N, sse_kind> va(a, aligned_t());
	simd_vec<T, N, sse_kind> vb(b, aligned_t());

	ASSERT_SIMD_EQ( va % vb, r);

	va %= vb;
	ASSERT_SIMD_EQ( va, r);
}


GCASE1( scale )
{
	LSIMD_ALIGN_SSE T a[MaxVLen] = { T(1), T(2), T(3), T(4) };
	T b = 2.5;
	simd_pack<T, sse_kind> bv( b );

	T r[N];
	for (int i = 0; i < N; ++i) r[i] = a[i] * b;

	simd_vec<T, N, sse_kind> va(a, aligned_t());

	ASSERT_SIMD_EQ( va * bv, r );

	va *= bv;
	ASSERT_SIMD_EQ( va, r);
}




GCASE1( sum )
{
	LSIMD_ALIGN_SSE T a[MaxVLen] = { T(1), T(2), T(3), T(4) };

	T s(0);
	for (int i = 0; i < N; ++i) s += a[i];

	simd_vec<T, N, sse_kind> va(a, aligned_t());
	ASSERT_EQ( va.sum(), s );
}


GCASE1( dot )
{
	LSIMD_ALIGN_SSE T a[MaxVLen] = { T(1), T(2), T(3), T(4) };
	LSIMD_ALIGN_SSE T b[MaxVLen] = { T(3), T(7), T(2), T(6) };

	T s(0);
	for (int i = 0; i < N; ++i) s += a[i] * b[i];

	simd_vec<T, N, sse_kind> va(a, aligned_t());
	simd_vec<T, N, sse_kind> vb(b, aligned_t());

	ASSERT_EQ( va.dot(vb), s );
}


template<template<typename T, int N> class H>
test_pack* make_tpack( const char *name )
{
	test_pack *tp = new test_pack( name );

	tp->add( new H<f32, 1>() );
	tp->add( new H<f32, 2>() );
	tp->add( new H<f32, 3>() );
	tp->add( new H<f32, 4>() );

	tp->add( new H<f64, 1>() );
	tp->add( new H<f64, 2>() );
	tp->add( new H<f64, 3>() );
	tp->add( new H<f64, 4>() );

	return tp;
}

#define ADD_TEST( name ) lsimd_main_suite.add( make_tpack<name##_tests>( #name ) )

void lsimd::add_test_packs()
{
	ADD_TEST( zero );
	ADD_TEST( set );
	ADD_TEST( load );
	ADD_TEST( store );
	ADD_TEST( bsxp );

	ADD_TEST( add );
	ADD_TEST( sub );
	ADD_TEST( mul );
	ADD_TEST( scale );

	ADD_TEST( sum );
	ADD_TEST( dot );

}




