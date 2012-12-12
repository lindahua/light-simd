/**
 * @file test_sse_packs.cpp
 *
 * Testing the correctness of sse_pack classes
 *
 * @author Dahua Lin
 */


#include "test_aux.h"
#include <light_simd/sse/sse_pack.h>

using namespace lsimd;
using namespace ltest;

// explicit instantiation for thorough syntax check


template struct lsimd::simd_pack<f32, sse_kind>;
template struct lsimd::simd_pack<f64, sse_kind>;

/************************************************
 *
 *  constructs
 *
 ************************************************/

GCASE( zero )
{
	T r[4] = {T(0), T(0), T(0), T(0)};

	simd_pack<T, sse_kind> p = tag::all_zeros();
	ASSERT_SIMD_EQ( p, r );
}

GCASE( load )
{
	LSIMD_ALIGN_SSE T a[5] = {T(1), T(2), T(3), T(5), T(4)};

	simd_pack<T, sse_kind> p;
	p.load(a, tag::aligned());
	ASSERT_SIMD_EQ( p, a );

	p.load(a + 1, tag::unaligned());
	ASSERT_SIMD_EQ( p, a + 1 );

	simd_pack<T, sse_kind> pa( a, tag::aligned() );
	ASSERT_SIMD_EQ( pa, a );

	simd_pack<T, sse_kind> pu( a + 1, tag::unaligned() );
	ASSERT_SIMD_EQ( pu, a + 1 );
}

GCASE( store )
{
	const int w = (int)simd_traits<T, sse_kind>::pack_width;

	LSIMD_ALIGN_SSE T s[4] = {T(1), T(3), T(2), T(4)};
	LSIMD_ALIGN_SSE T t[5];

	simd_pack<T, sse_kind> p(s, tag::aligned());
	ASSERT_SIMD_EQ( p, s );

	clear_zeros(5, t);
	p.store(t, tag::aligned() );
	ASSERT_VEC_EQ(w, t, s);

	clear_zeros(5, t);
	p.store(t+1, tag::unaligned());
	ASSERT_VEC_EQ(w, t+1, s);
}

template<typename T> class partial_load_tests;

SCASE( partial_load, f32 )
{
	f32 a[4] = {1.f, 2.f, 3.f, 4.f};

	simd_pack<f32, sse_kind> v;

	v.partial_load(a, int_<0>());
	f32 r0[4] = {0.f, 0.f, 0.f, 0.f};
	ASSERT_SIMD_EQ(v, r0);

	v.partial_load(a, int_<1>());
	f32 r1[4] = {1.f, 0.f, 0.f, 0.f};
	ASSERT_SIMD_EQ(v, r1);

	v.partial_load(a, int_<2>());
	f32 r2[4] = {1.f, 2.f, 0.f, 0.f};
	ASSERT_SIMD_EQ(v, r2);

	v.partial_load(a, int_<3>());
	f32 r3[4] = {1.f, 2.f, 3.f, 0.f};
	ASSERT_SIMD_EQ(v, r3);

	v.partial_load(a, int_<4>());
	f32 r4[4] = {1.f, 2.f, 3.f, 4.f};
	ASSERT_SIMD_EQ(v, r4);
}


SCASE( partial_load, f64 )
{
	f64 a[2] = {1.0, 2.0};

	simd_pack<f64, sse_kind> v;

	v.partial_load(a, int_<0>());
	f64 r0[2] = {0.0, 0.0};
	ASSERT_SIMD_EQ(v, r0);

	v.partial_load(a, int_<1>());
	f64 r1[2] = {1.0, 0.0};
	ASSERT_SIMD_EQ(v, r1);

	v.partial_load(a, int_<2>());
	f64 r2[2] = {1.0, 2.0};
	ASSERT_SIMD_EQ(v, r2);
}


template<typename T> class partial_store_tests;

SCASE( partial_store, f32 )
{
	f32 a[4] = {1.f, 2.f, 3.f, 4.f};
	f32 b[4];

	simd_pack<f32, sse_kind> p(a, tag::unaligned());

	for (int i = 0; i < 4; ++i) b[i] = -1.f;
	p.partial_store(b, int_<0>());
	f32 r0[4] = {-1.f, -1.f, -1.f, -1.f};
	ASSERT_VEC_EQ(4, b, r0);

	for (int i = 0; i < 4; ++i) b[i] = -1.f;
	p.partial_store(b, int_<1>());
	f32 r1[4] = {1.f, -1.f, -1.f, -1.f};
	ASSERT_VEC_EQ(4, b, r1);

	for (int i = 0; i < 4; ++i) b[i] = -1.f;
	p.partial_store(b, int_<2>());
	f32 r2[4] = {1.f, 2.f, -1.f, -1.f};
	ASSERT_VEC_EQ(4, b, r2);

	for (int i = 0; i < 4; ++i) b[i] = -1.f;
	p.partial_store(b, int_<3>());
	f32 r3[4] = {1.f, 2.f, 3.f, -1.f};
	ASSERT_VEC_EQ(4, b, r3);

	for (int i = 0; i < 4; ++i) b[i] = -1.f;
	p.partial_store(b, int_<4>());
	f32 r4[4] = {1.f, 2.f, 3.f, 4.f};
	ASSERT_VEC_EQ(4, b, r4);
}

SCASE( partial_store, f64 )
{
	f64 a[2] = {1.0, 2.0};
	f64 b[2] = {-1.0, -1.0};

	simd_pack<f64, sse_kind> p(a, tag::unaligned());

	p.partial_store(b, int_<0>());
	f64 r0[2] = {-1.0, -1.0};
	ASSERT_VEC_EQ(2, b, r0);

	p.partial_store(b, int_<1>());
	f64 r1[2] = {1.0, -1.0};
	ASSERT_VEC_EQ(2, b, r1);

	p.partial_store(b, int_<2>());
	f64 r2[2] = {1.0, 2.0};
	ASSERT_VEC_EQ(2, b, r2);
}



template<typename T> class set_tests;

SCASE( set, f32 )
{
	const f32 v1(1.23f);
	const f32 v2(-3.42f);
	const f32 v3(4.57f);
	const f32 v4(-5.26f);

	LSIMD_ALIGN_SSE f32 r1[4] = {v1, v1, v1, v1};
	LSIMD_ALIGN_SSE f32 r2[4] = {v1, v2, v3, v4};

	simd_pack<f32, sse_kind> p(v1);
	ASSERT_SIMD_EQ( p, r1 );

	simd_pack<f32, sse_kind> p2;
	p2.set(v1);
	ASSERT_SIMD_EQ( p, r1 );

	sse_f32pk q(v1, v2, v3, v4);
	ASSERT_TRUE( q.test_equal(r2) );

	sse_f32pk q2;
	q2.set(v1, v2, v3, v4);
	ASSERT_TRUE( q2.test_equal(r2) );
}


SCASE( set, f64 )
{
	const f64 v1(1.23);
	const f64 v2(-3.42);

	LSIMD_ALIGN_SSE f64 r1[2] = {v1, v1};
	LSIMD_ALIGN_SSE f64 r2[2] = {v1, v2};

	simd_pack<f64, sse_kind> p(v1);
	ASSERT_SIMD_EQ( p, r1 );

	simd_pack<f64, sse_kind> p2;
	p2.set(v1);
	ASSERT_SIMD_EQ( p2, r1 );

	sse_f64pk q(v1, v2);
	ASSERT_TRUE( q.test_equal(r2) );

	sse_f64pk q2;
	q2.set(v1, v2);
	ASSERT_TRUE( q2.test_equal(r2) );
}


test_pack* tpack_constructs()
{
	ltest::test_pack* tp = new test_pack("constructs");

	tp->add( new zero_tests<f32>() );
	tp->add( new zero_tests<f64>() );

	tp->add( new load_tests<f32>() );
	tp->add( new load_tests<f64>() );

	tp->add( new store_tests<f32>() );
	tp->add( new store_tests<f64>() );

	tp->add( new partial_load_tests<f32>() );
	tp->add( new partial_load_tests<f64>() );

	tp->add( new partial_store_tests<f32>() );
	tp->add( new partial_store_tests<f64>() );

	tp->add( new set_tests<f32>() );
	tp->add( new set_tests<f64>() );

	return tp;
}


/************************************************
 *
 *  entry manipulation
 *
 ************************************************/


GCASE( to_scalar )
{
	T sv = T(1.25);
	LSIMD_ALIGN_SSE T src[4] = {sv, T(1), T(2), T(3)};

	simd_pack<T> a(src, tag::aligned());

	ASSERT_EQ( a.to_scalar(), sv );
}


template<typename T> class extract_tests;

SCASE( extract, f32 )
{
	LSIMD_ALIGN_SSE f32 src[4] = {1.11f, 2.22f, 3.33f, 4.44f};

	simd_pack<f32, sse_kind> a(src, tag::aligned());

	f32 e0 = a.extract<0>();
	f32 e1 = a.extract<1>();
	f32 e2 = a.extract<2>();
	f32 e3 = a.extract<3>();

	ASSERT_EQ( e0, src[0]);
	ASSERT_EQ( e1, src[1]);
	ASSERT_EQ( e2, src[2]);
	ASSERT_EQ( e3, src[3] );
}


SCASE( extract, f64 )
{
	LSIMD_ALIGN_SSE f64 src[4] = {1.11, 2.22, 3.33, 4.44};

	simd_pack<f64, sse_kind> a(src, tag::aligned());

	f64 e0 = a.extract<0>();
	f64 e1 = a.extract<1>();

	ASSERT_EQ( e0, src[0] );
	ASSERT_EQ( e1, src[1] );
}


template<typename T> class broadcast_tests;

SCASE( broadcast, f32 )
{
	LSIMD_ALIGN_SSE f32 s[4] = {1.f, 2.f, 3.f, 4.f};

	simd_pack<f32, sse_kind> a(s, tag::aligned());

	LSIMD_ALIGN_SSE f32 r0[4] = {s[0], s[0], s[0], s[0]};
	ASSERT_SIMD_EQ( broadcast(a, int_<0>()), r0 );

	LSIMD_ALIGN_SSE f32 r1[4] = {s[1], s[1], s[1], s[1]};
	ASSERT_SIMD_EQ( broadcast(a, int_<1>()), r1 );

	LSIMD_ALIGN_SSE f32 r2[4] = {s[2], s[2], s[2], s[2]};
	ASSERT_SIMD_EQ( broadcast(a, int_<2>()), r2 );

	LSIMD_ALIGN_SSE f32 r3[4] = {s[3], s[3], s[3], s[3]};
	ASSERT_SIMD_EQ( broadcast(a, int_<3>()), r3 );
}


SCASE( broadcast, f64 )
{
	LSIMD_ALIGN_SSE f64 s[2] = {1.0, 2.0};

	simd_pack<f64, sse_kind> a(s, tag::aligned());

	LSIMD_ALIGN_SSE f64 r0[4] = {s[0], s[0]};
	ASSERT_SIMD_EQ( broadcast(a, int_<0>()), r0 );

	LSIMD_ALIGN_SSE f64 r1[4] = {s[1], s[1]};
	ASSERT_SIMD_EQ( broadcast(a, int_<1>()), r1 );
}



template<typename T> class duplicate_tests;

SCASE( duplicate, f32 )
{
	LSIMD_ALIGN_SSE f32 s[4] = {1.f, 2.f, 3.f, 4.f};

	sse_f32pk a(s, tag::aligned());

	LSIMD_ALIGN_SSE f32 r01[4] = {1.f, 1.f, 2.f, 2.f};
	ASSERT_SIMD_EQ( swizzle(a, pat4_<0, 0, 1, 1>()), r01);

	LSIMD_ALIGN_SSE f32 r23[4] = {3.f, 3.f, 4.f, 4.f};
	ASSERT_SIMD_EQ( swizzle(a, pat4_<2, 2, 3, 3>()), r23);

	LSIMD_ALIGN_SSE f32 r02[4] = {1.f, 1.f, 3.f, 3.f};
	ASSERT_SIMD_EQ( swizzle(a, pat4_<0, 0, 2, 2>()), r02);

	LSIMD_ALIGN_SSE f32 r13[4] = {2.f, 2.f, 4.f, 4.f};
	ASSERT_SIMD_EQ( swizzle(a, pat4_<1, 1, 3, 3>()), r13);

	LSIMD_ALIGN_SSE f32 s01[4] = {1.f, 2.f, 1.f, 2.f};
	ASSERT_SIMD_EQ( swizzle(a, pat4_<0, 1, 0, 1>()), s01);

	LSIMD_ALIGN_SSE f32 s23[4] = {3.f, 4.f, 3.f, 4.f};
	ASSERT_SIMD_EQ( swizzle(a, pat4_<2, 3, 2, 3>()), s23);
}


SCASE( duplicate, f64 )
{
	LSIMD_ALIGN_SSE f64 s[2] = {1.0, 2.0};

	sse_f64pk a(s, tag::aligned());

	LSIMD_ALIGN_SSE f64 r0[2] = {1.0, 1.0};
	ASSERT_SIMD_EQ( swizzle(a, pat2_<0, 0>()), r0);

	LSIMD_ALIGN_SSE f64 r1[2] = {2.0, 2.0};
	ASSERT_SIMD_EQ( swizzle(a, pat2_<1, 1>()), r1);
}


template<typename T> class swizzle_tests;

SCASE( swizzle, f32 )
{
	LSIMD_ALIGN_SSE f32 s[4] = {1.f, 2.f, 3.f, 4.f};

	sse_f32pk a(s, tag::aligned());

	LSIMD_ALIGN_SSE f32 r1[4] = {1.f, 2.f, 3.f, 4.f};
	ASSERT_SIMD_EQ( swizzle(a, pat4_<0,1,2,3>()), r1);

	LSIMD_ALIGN_SSE f32 r2[4] = {4.f, 3.f, 2.f, 1.f};
	ASSERT_SIMD_EQ( swizzle(a, pat4_<3,2,1,0>()), r2);

	LSIMD_ALIGN_SSE f32 r3[4] = {2.f, 1.f, 4.f, 3.f};
	ASSERT_SIMD_EQ( swizzle(a, pat4_<1,0,3,2>()), r3);

	LSIMD_ALIGN_SSE f32 r4[4] = {3.f, 4.f, 1.f, 2.f};
	ASSERT_SIMD_EQ( swizzle(a, pat4_<2,3,0,1>()), r4);

	LSIMD_ALIGN_SSE f32 r5[4] = {3.f, 2.f, 1.f, 4.f};
	ASSERT_SIMD_EQ( swizzle(a, pat4_<2,1,0,3>()), r5);
}


SCASE( swizzle, f64 )
{
	LSIMD_ALIGN_SSE f64 s[4] = {1.0, 2.0};

	sse_f64pk a(s, tag::aligned());

	LSIMD_ALIGN_SSE f64 r0[2] = {1.0, 1.0};
	ASSERT_SIMD_EQ( swizzle(a, pat2_<0,0>()), r0);

	LSIMD_ALIGN_SSE f64 r1[2] = {1.0, 2.0};
	ASSERT_SIMD_EQ( swizzle(a, pat2_<0,1>()), r1);

	LSIMD_ALIGN_SSE f64 r2[2] = {2.0, 1.0};
	ASSERT_SIMD_EQ( swizzle(a, pat2_<1,0>()), r2);

	LSIMD_ALIGN_SSE f64 r3[2] = {2.0, 2.0};
	ASSERT_SIMD_EQ( swizzle(a, pat2_<1,1>()), r3);
}

test_pack* tpack_manipulates()
{
	ltest::test_pack* tp = new test_pack("manipulates");

	tp->add( new to_scalar_tests<f32>() );
	tp->add( new to_scalar_tests<f64>() );

	tp->add( new extract_tests<f32>() );
	tp->add( new extract_tests<f64>() );

	tp->add( new broadcast_tests<f32>() );
	tp->add( new broadcast_tests<f64>() );

	tp->add( new duplicate_tests<f32>() );
	tp->add( new duplicate_tests<f64>() );

	tp->add( new swizzle_tests<f32>() );
	tp->add( new swizzle_tests<f64>() );

	return tp;
}


void lsimd::add_test_packs()
{
	lsimd_main_suite.add( tpack_constructs() );
	lsimd_main_suite.add( tpack_manipulates() );
}





