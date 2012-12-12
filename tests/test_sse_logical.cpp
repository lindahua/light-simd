/*
 * @file test_sse_logical.cpp
 *
 * Unit testing of SSE comparison and logical functions
 *
 * @author Dahua Lin
 */


#include "test_aux.h"
#include <light_simd/sse/sse_logical.h>

#include <cstring>

using namespace lsimd;
using namespace ltest;

template<typename T>
struct bool_pack;

const int FALSE_MSK[4] = { 0, 0, 0, 0 };
const int TRUE_MSK[4] = { (int)(0xffffffff), (int)(0xffffffff), (int)(0xffffffff), (int)(0xffffffff) };

template<typename T>
inline bool is_false_mask(T v)
{
	return std::memcmp(&v, FALSE_MSK, sizeof(T)) == 0;
}

template<typename T>
inline bool is_true_mask(T v)
{
	return std::memcmp(&v, TRUE_MSK, sizeof(T)) == 0;
}

template<typename T>
inline bool is_mask_eq(T v, bool b)
{
	return b ? is_true_mask(v) : is_false_mask(v);
}


template<typename T>
inline bool is_mask_eq(const simd_pack<T, sse_kind>&  p, const bool *ba)
{
	const unsigned w = simd_traits<T, sse_kind>::pack_width;

	for (unsigned i = 0; i < w; ++i)
	{
		if (!is_mask_eq(p.e[i], ba[i])) return false;
	}
	return true;
}



GCASE( consts )
{
	simd_pack<T, sse_kind> tm = tag::all_nonzeros();
	simd_pack<T, sse_kind> fm = tag::all_zeros();

	bool ta[4] = {true, true, true, true};
	bool fa[4] = {false, false, false, false};

	ASSERT_TRUE( is_mask_eq(tm, ta) );
	ASSERT_TRUE( is_mask_eq(fm, fa) );
}

template<typename T> class constructs_tests;

SCASE( constructs, f32 )
{
	const bool ba[4] = {true, false, true, false};
	simd_pack<f32, sse_kind> p( sse_f32pk(ba[0], ba[1], ba[2], ba[3]) );
	ASSERT_TRUE( is_mask_eq(p, ba) );
}

SCASE( constructs, f64 )
{
	const bool ba[2] = {true, false};
	simd_pack<f64, sse_kind> p( sse_f64pk(ba[0], ba[1]) );
	ASSERT_TRUE( is_mask_eq(p, ba) );
}


GCASE( cmp_eq )
{
	LSIMD_ALIGN_SSE T a[4] = {T(1), T(2), T(3), T(4)};
	LSIMD_ALIGN_SSE T b[4] = {T(2), T(1), T(4), T(3)};
	LSIMD_ALIGN_SSE T c[4] = {T(1), T(1), T(3), T(3)};

	bool ab[4], ac[4];
	for (int i = 0; i < 4; ++i)
	{
		ab[i] = (a[i] == b[i]);
		ac[i] = (a[i] == c[i]);
	}

	simd_pack<T, sse_kind> pa(a, tag::aligned());
	simd_pack<T, sse_kind> pb(b, tag::aligned());
	simd_pack<T, sse_kind> pc(c, tag::aligned());

	ASSERT_TRUE( is_mask_eq(pa == pb, ab) );
	ASSERT_TRUE( is_mask_eq(pa == pc, ac) );
}

GCASE( cmp_ne )
{
	LSIMD_ALIGN_SSE T a[4] = {T(1), T(2), T(3), T(4)};
	LSIMD_ALIGN_SSE T b[4] = {T(2), T(1), T(4), T(3)};
	LSIMD_ALIGN_SSE T c[4] = {T(1), T(1), T(3), T(3)};

	bool ab[4], ac[4];
	for (int i = 0; i < 4; ++i)
	{
		ab[i] = (a[i] != b[i]);
		ac[i] = (a[i] != c[i]);
	}

	simd_pack<T, sse_kind> pa(a, tag::aligned());
	simd_pack<T, sse_kind> pb(b, tag::aligned());
	simd_pack<T, sse_kind> pc(c, tag::aligned());

	ASSERT_TRUE( is_mask_eq(pa != pb, ab) );
	ASSERT_TRUE( is_mask_eq(pa != pc, ac) );
}

GCASE( cmp_lt )
{
	LSIMD_ALIGN_SSE T a[4] = {T(1), T(2), T(3), T(4)};
	LSIMD_ALIGN_SSE T b[4] = {T(2), T(1), T(4), T(3)};
	LSIMD_ALIGN_SSE T c[4] = {T(1), T(1), T(3), T(3)};

	bool ab[4], ac[4];
	for (int i = 0; i < 4; ++i)
	{
		ab[i] = a[i] < b[i];
		ac[i] = a[i] < c[i];
	}

	simd_pack<T, sse_kind> pa(a, tag::aligned());
	simd_pack<T, sse_kind> pb(b, tag::aligned());
	simd_pack<T, sse_kind> pc(c, tag::aligned());

	ASSERT_TRUE( is_mask_eq(pa < pb, ab) );
	ASSERT_TRUE( is_mask_eq(pa < pc, ac) );
}

GCASE( cmp_le )
{
	LSIMD_ALIGN_SSE T a[4] = {T(1), T(2), T(3), T(4)};
	LSIMD_ALIGN_SSE T b[4] = {T(2), T(1), T(4), T(3)};
	LSIMD_ALIGN_SSE T c[4] = {T(1), T(1), T(3), T(3)};

	bool ab[4], ac[4];
	for (int i = 0; i < 4; ++i)
	{
		ab[i] = a[i] <= b[i];
		ac[i] = a[i] <= c[i];
	}

	simd_pack<T, sse_kind> pa(a, tag::aligned());
	simd_pack<T, sse_kind> pb(b, tag::aligned());
	simd_pack<T, sse_kind> pc(c, tag::aligned());

	ASSERT_TRUE( is_mask_eq(pa <= pb, ab) );
	ASSERT_TRUE( is_mask_eq(pa <= pc, ac) );
}

GCASE( cmp_gt )
{
	LSIMD_ALIGN_SSE T a[4] = {T(1), T(2), T(3), T(4)};
	LSIMD_ALIGN_SSE T b[4] = {T(2), T(1), T(4), T(3)};
	LSIMD_ALIGN_SSE T c[4] = {T(1), T(1), T(3), T(3)};

	bool ab[4], ac[4];
	for (int i = 0; i < 4; ++i)
	{
		ab[i] = a[i] > b[i];
		ac[i] = a[i] > c[i];
	}

	simd_pack<T, sse_kind> pa(a, tag::aligned());
	simd_pack<T, sse_kind> pb(b, tag::aligned());
	simd_pack<T, sse_kind> pc(c, tag::aligned());

	ASSERT_TRUE( is_mask_eq(pa > pb, ab) );
	ASSERT_TRUE( is_mask_eq(pa > pc, ac) );
}

GCASE( cmp_ge )
{
	LSIMD_ALIGN_SSE T a[4] = {T(1), T(2), T(3), T(4)};
	LSIMD_ALIGN_SSE T b[4] = {T(2), T(1), T(4), T(3)};
	LSIMD_ALIGN_SSE T c[4] = {T(1), T(1), T(3), T(3)};

	bool ab[4], ac[4];
	for (int i = 0; i < 4; ++i)
	{
		ab[i] = a[i] >= b[i];
		ac[i] = a[i] >= c[i];
	}

	simd_pack<T, sse_kind> pa(a, tag::aligned());
	simd_pack<T, sse_kind> pb(b, tag::aligned());
	simd_pack<T, sse_kind> pc(c, tag::aligned());

	ASSERT_TRUE( is_mask_eq(pa >= pb, ab) );
	ASSERT_TRUE( is_mask_eq(pa >= pc, ac) );
}



template<typename T> class bitwise_not_tests;

SCASE( bitwise_not, f32 )
{
	simd_pack<f32, sse_kind> p( sse_f32pk(false, true, false, true) );
	const bool ba[4] = {true, false, true, false};

	ASSERT_TRUE( is_mask_eq(~p, ba) );
}

SCASE( bitwise_not, f64 )
{
	simd_pack<f64, sse_kind> p( sse_f64pk(false, true) );
	const bool ba[2] = {true, false};

	ASSERT_TRUE( is_mask_eq(~p, ba) );
}


template<typename T> class bitwise_and_tests;

SCASE( bitwise_and, f32 )
{
	simd_pack<f32, sse_kind> p( sse_f32pk(false, false, true, true) );
	simd_pack<f32, sse_kind> q( sse_f32pk(false, true, false, true) );

	const bool ba[4] = {false, false, false, true};
	ASSERT_TRUE( is_mask_eq(p & q, ba) );
}

SCASE( bitwise_and, f64 )
{
	simd_pack<f64, sse_kind> f( sse_f64pk(false, false) );
	simd_pack<f64, sse_kind> t( sse_f64pk(true, true) );

	simd_pack<f64, sse_kind> p( sse_f64pk(false, true) );

	const bool ba0[2] = {false, false};
	const bool ba1[2] = {false, true};

	ASSERT_TRUE( is_mask_eq(f & p, ba0) );
	ASSERT_TRUE( is_mask_eq(t & p, ba1) );
}


template<typename T> class bitwise_or_tests;

SCASE( bitwise_or, f32 )
{
	simd_pack<f32, sse_kind> p( sse_f32pk(false, false, true, true) );
	simd_pack<f32, sse_kind> q( sse_f32pk(false, true, false, true) );

	const bool ba[4] = {false, true, true, true};
	ASSERT_TRUE( is_mask_eq(p | q, ba) );
}

SCASE( bitwise_or, f64 )
{
	simd_pack<f64, sse_kind> f( sse_f64pk(false, false) );
	simd_pack<f64, sse_kind> t( sse_f64pk(true, true) );

	simd_pack<f64, sse_kind> p( sse_f64pk(false, true) );

	const bool ba0[2] = {false, true};
	const bool ba1[2] = {true, true};

	ASSERT_TRUE( is_mask_eq(f | p, ba0) );
	ASSERT_TRUE( is_mask_eq(t | p, ba1) );
}


template<typename T> class bitwise_xor_tests;

SCASE( bitwise_xor, f32 )
{
	simd_pack<f32, sse_kind> p( sse_f32pk(false, false, true, true) );
	simd_pack<f32, sse_kind> q( sse_f32pk(false, true, false, true) );

	const bool ba[4] = {false, true, true, false};
	ASSERT_TRUE( is_mask_eq(p ^ q, ba) );
}

SCASE( bitwise_xor, f64 )
{
	simd_pack<f64, sse_kind> f( sse_f64pk(false, false) );
	simd_pack<f64, sse_kind> t( sse_f64pk(true, true) );

	simd_pack<f64, sse_kind> p( sse_f64pk(false, true) );

	const bool ba0[2] = {false, true};
	const bool ba1[2] = {true, false};

	ASSERT_TRUE( is_mask_eq(f ^ p, ba0) );
	ASSERT_TRUE( is_mask_eq(t ^ p, ba1) );
}

GCASE( cond_select_sse2 )
{
	const bool msk[4] = { false, true, false, true };
	LSIMD_ALIGN_SSE T mskv[4] = {T(0), T(1), T(0), T(1)};

	LSIMD_ALIGN_SSE T x[4] = { T(1), T(2), T(3), T(4) };
	LSIMD_ALIGN_SSE T y[4] = { T(11), T(22), T(33), T(44) };
	LSIMD_ALIGN_SSE T r[4];

	for (int i = 0; i < 4; ++i)
	{
		r[i] = (msk[i] ? x[i] : y[i]);
	}

	simd_pack<T, sse_kind> pmv(mskv, tag::aligned());
	simd_pack<T, sse_kind> pzero = tag::all_zeros();
	simd_pack<T, sse_kind> pm = pmv > pzero;

	simd_pack<T, sse_kind> px(x, tag::aligned());
	simd_pack<T, sse_kind> py(y, tag::aligned());

	simd_pack<T, sse_kind> pr( cond_sse2(pm, px, py) );
	ASSERT_SIMD_EQ( pr, r );
}

GCASE( cond_select )
{
	const bool msk[4] = { false, true, false, true };
	LSIMD_ALIGN_SSE T mskv[4] = {T(0), T(1), T(0), T(1)};

	LSIMD_ALIGN_SSE T x[4] = { T(1), T(2), T(3), T(4) };
	LSIMD_ALIGN_SSE T y[4] = { T(11), T(22), T(33), T(44) };
	LSIMD_ALIGN_SSE T r[4];

	for (int i = 0; i < 4; ++i)
	{
		r[i] = (msk[i] ? x[i] : y[i]);
	}

	simd_pack<T, sse_kind> pmv(mskv, tag::aligned());

	simd_pack<T, sse_kind> pzero = tag::all_zeros();
	simd_pack<T, sse_kind> pm = pmv > pzero;

	simd_pack<T, sse_kind> px(x, tag::aligned());
	simd_pack<T, sse_kind> py(y, tag::aligned());

	ASSERT_SIMD_EQ( cond(pm, px, py), r );
}


test_pack* logical_tpack_constructs()
{
	test_pack *tp = new test_pack( "mask_constructs" );
	tp->add( new consts_tests<f32>() );
	tp->add( new consts_tests<f64>() );
	tp->add( new constructs_tests<f32>() );
	tp->add( new constructs_tests<f64>() );
	return tp;
}

test_pack* logical_tpack_compare()
{
	test_pack *tp = new test_pack( "logical_compare" );
	tp->add( new cmp_eq_tests<f32>() );
	tp->add( new cmp_eq_tests<f64>() );
	tp->add( new cmp_ne_tests<f32>() );
	tp->add( new cmp_ne_tests<f64>() );
	tp->add( new cmp_lt_tests<f32>() );
	tp->add( new cmp_lt_tests<f64>() );
	tp->add( new cmp_le_tests<f32>() );
	tp->add( new cmp_le_tests<f64>() );
	tp->add( new cmp_gt_tests<f32>() );
	tp->add( new cmp_gt_tests<f64>() );
	tp->add( new cmp_ge_tests<f32>() );
	tp->add( new cmp_ge_tests<f64>() );
	return tp;
}

test_pack* logical_tpack_bitops()
{
	test_pack *tp = new test_pack( "logical_bitops" );
	tp->add( new bitwise_not_tests<f32>() );
	tp->add( new bitwise_not_tests<f64>() );
	tp->add( new bitwise_and_tests<f32>() );
	tp->add( new bitwise_and_tests<f64>() );
	tp->add( new bitwise_or_tests<f32>() );
	tp->add( new bitwise_or_tests<f64>() );
	tp->add( new bitwise_xor_tests<f32>() );
	tp->add( new bitwise_xor_tests<f64>() );
	return tp;
}


test_pack* logical_tpack_condsel()
{
	test_pack *tp = new test_pack( "cond_selects" );
	tp->add( new cond_select_sse2_tests<f32>() );
	tp->add( new cond_select_sse2_tests<f64>() );
	tp->add( new cond_select_tests<f32>() );
	tp->add( new cond_select_tests<f64>() );
	return tp;
}


void lsimd::add_test_packs()
{
	lsimd_main_suite.add( logical_tpack_constructs() );
	lsimd_main_suite.add( logical_tpack_compare() );
	lsimd_main_suite.add( logical_tpack_bitops() );
	lsimd_main_suite.add( logical_tpack_condsel() );
}








