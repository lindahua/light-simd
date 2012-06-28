/*
 * @file sse_mat_sol_bits.h
 *
 * The internal implementation of matrix inverse & equation solving
 *
 * @author Dahua Lin
 */

#ifndef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_MAT_SOL_BITS_H_
#define LSIMD_SSE_MAT_SOL_BITS_H_

#include "sse_mat_matmul_bits.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4141 4127)
#endif

namespace lsimd { namespace sse {

	/**********************************
	 *
	 *  Auxiliary functions
	 *
	 **********************************/

	LSIMD_ENSURE_INLINE
	inline __m128 low2_mask_f32()
	{
		return _mm_castsi128_ps(
				_mm_setr_epi32((int)0xffffffff, (int)0xffffffff, 0, 0));
	}

	LSIMD_ENSURE_INLINE
	inline __m128 low3_mask_f32()
	{
		return _mm_castsi128_ps(
				_mm_setr_epi32((int)0xffffffff, (int)0xffffffff, (int)0xffffffff, 0));
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk hdiff(const sse_f32pk& p)
	{
		return sub_s(p, p.dup2_high());
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk hdiff(const sse_f64pk& p)
	{
		return sub_s(p, p.dup_high());
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk rev_hdiff(const sse_f64pk& p)
	{
		return sub_s(p.dup_high(), p);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk pre_det(const smat_core<f32,2,2>& a)
	{
		return a.col01_pk * a.col01_pk.swizzle<3,2,1,0>();
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk pre_det(const smat_core<f64,2,2>& a)
	{
		return a.col0.m_pk * a.col1.m_pk.swizzle<1,0>();
	}

	LSIMD_ENSURE_INLINE
	inline smat_core<f32,2,2> mm2x2(const smat_core<f32,2,2>& a, const smat_core<f32,2,2>& b)
	{
		return  (a.col01_pk.dup_low()  * b.col01_pk.dup2_low()) +
				(a.col01_pk.dup_high() * b.col01_pk.dup2_high());
	}

	LSIMD_ENSURE_INLINE
	inline smat_core<f64,2,2> mm2x2(const smat_core<f64,2,2>& a, const smat_core<f64,2,2>& b)
	{
		smat_core<f64, 2, 2> r;
		r.col0 = a.col0 * b.col0.bsx_pk<0>() + a.col1 * b.col0.bsx_pk<1>();
		r.col1 = a.col0 * b.col1.bsx_pk<0>() + a.col1 * b.col1.bsx_pk<1>();
		return r;
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk mm2x2_trace_p(const smat_core<f32,2,2>& a, const smat_core<f32,2,2>& b)
	{
		sse_f32pk p = a.col01_pk * b.col01_pk.swizzle<0,2,1,3>();
		p = p + p.dup_high();
		return add_s(p, p.dup2_high());
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk mm2x2_trace_p(const smat_core<f64,2,2>& a, const smat_core<f64,2,2>& b)
	{
		sse_f64pk p = a.col0.m_pk * unpack_low(b.col0.m_pk, b.col1.m_pk);
		p = p + a.col1.m_pk * unpack_high(b.col0.m_pk, b.col1.m_pk);
		return add_s(p, p.dup_high());
	}

	LSIMD_ENSURE_INLINE
	inline smat_core<f32,2,2> adjoint_signmask_f32()
	{
		sse_f32pk p = _mm_castsi128_ps(
				_mm_setr_epi32(0, (int)0x80000000, (int)0x80000000, 0));
		return p;
	}

	LSIMD_ENSURE_INLINE
	inline smat_core<f32,2,2> adjoint_mat(const smat_core<f32,2,2>& a, const smat_core<f32,2,2>& mask)
	{
		sse_f32pk p = _mm_xor_ps(a.col01_pk.swizzle<3,1,2,0>().v, mask.col01_pk.v);
		return p;
	}

	LSIMD_ENSURE_INLINE
	inline smat_core<f64,2,2> adjoint_signmask_f64()
	{
		smat_core<f64,2,2> m;
		m.col0.m_pk = _mm_setr_pd(0.0, -0.0);
		m.col1.m_pk = _mm_setr_pd(-0.0, 0.0);
		return m;
	}

	LSIMD_ENSURE_INLINE
	inline smat_core<f64,2,2> adjoint_mat(const smat_core<f64,2,2>& a, const smat_core<f64,2,2>& mask)
	{
		smat_core<f64,2,2> r;
		r.col0.m_pk = _mm_xor_pd(unpack_high(a.col1.m_pk, a.col0.m_pk).v, mask.col0.m_pk.v);
		r.col1.m_pk = _mm_xor_pd(unpack_low (a.col1.m_pk, a.col0.m_pk).v, mask.col1.m_pk.v);
		return r;
	}


	/**********************************
	 *
	 *  Matrix 2 x 2
	 *
	 **********************************/

	LSIMD_ENSURE_INLINE
	inline f32 det(const smat_core<f32, 2, 2>& a)
	{
		sse_f32pk p = pre_det(a);
		return hdiff(p).to_scalar();
	}

	LSIMD_ENSURE_INLINE
	inline f64 det(const smat_core<f64, 2, 2>& a)
	{
		sse_f64pk p = pre_det(a);
		return hdiff(p).to_scalar();
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk adjoint(const smat_core<f32, 2, 2>& a, smat_core<f32, 2, 2>& r)
	{
		r = adjoint_mat(a, adjoint_signmask_f32());

		return hdiff(pre_det(a));
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk adjoint(const smat_core<f64, 2, 2>& a, smat_core<f64, 2, 2>& r)
	{
		r = adjoint_mat(a, adjoint_signmask_f64());

		return hdiff(pre_det(a));
	}


	/**********************************
	 *
	 *  Matrix 3 x 3
	 *
	 **********************************/

	LSIMD_ENSURE_INLINE
	inline f32 det(const smat_core<f32, 3, 3>& a)
	{
		// generate product terms

		sse_f32pk b = shuffle<1,0,1,0>(a.col2.m_pk, a.col1.m_pk);

		sse_f32pk u1 = a.col0.m_pk.dup_low() * b;
		sse_f32pk u2 = a.col1.m_pk * b;

		u1 = u1 * shuffle<2,2,2,2>(a.col1.m_pk, a.col2.m_pk);
		u2 = u2 * shuffle<2,2,2,2>(a.col0.m_pk, a.col0.m_pk);

		// aggregate the terms

		u1 = u1.dup_high() - u1;
		u1 = u1 + u2;

		return hdiff(u1).to_scalar();
	}

	LSIMD_ENSURE_INLINE
	inline f64 det(const smat_core<f64, 3, 3>& a)
	{
		// generate product terms

		sse_f64pk c = a.col2.m_pk0.swizzle<1,0>();

		sse_f64pk u1 = a.col0.m_pk0 * a.col1.m_pk0.swizzle<1,0>();
		sse_f64pk u2 = a.col0.m_pk0 * c;
		sse_f64pk u3 = a.col1.m_pk0 * c;

		u1 = u1 * a.col2.m_pk1.bsx<0>();
		u2 = u2 * a.col1.m_pk1.bsx<0>();
		u3 = u3 * a.col0.m_pk1.bsx<0>();

		// aggregate the terms

		u1 = u1 - u2;
		u1 = u1 + u3;

		return hdiff(u1).to_scalar();
	}


	inline sse_f32pk adjoint(const smat_core<f32, 3, 3>& a, smat_core<f32, 3, 3>& r)
	{
		// calculate co-factor matrix

		sse_f32pk p0 = a.col0.m_pk.swizzle<1,2,0,2>();
		sse_f32pk p1 = a.col1.m_pk.swizzle<1,2,0,2>();

		sse_f32pk q1 = a.col1.m_pk.swizzle<2,1,2,0>();
		sse_f32pk q2 = a.col2.m_pk.swizzle<2,1,2,0>();

		sse_f32pk c0 = p1 * q2;
		sse_f32pk c1 = p0 * q2;
		sse_f32pk c2 = p0 * q1;

		c0 = shuffle<0,3,1,2>(c0, c1) - shuffle<1,2,0,3>(c0, c1);

		p0 = merge_low(a.col1.m_pk, a.col0.m_pk);
		q2 = a.col2.m_pk.swizzle<1,0,1,0>();

		sse_f32pk d0 = p0 * q2;

		c1 = shuffle<0,3,0,3>(c2, d0) - shuffle<1,2,1,2>(c2, d0);

		c2 = p0 * p0.swizzle<3,2,1,0>();
		c2 = sub_s(c2.shift_front<1>(), c2);

		// calculate determinant

		sse_f32pk detv = c0 * a.col0.m_pk;
		sse_f32pk v2 = mul_s(c1.dup_high(), a.col0.m_pk.dup_high());

		detv = add_s(add_s(detv, detv.dup2_high()), v2);

		// joint into adjoint matrix

		__m128 msk = _mm_castsi128_ps(
				_mm_setr_epi32((int)(0xffffffff), (int)(0xffffffff), (int)(0xffffffff), 0));

		r.col0.m_pk.v = _mm_and_ps(shuffle<0, 2, 0, 0>(c0, c1).v, msk);
		r.col1.m_pk.v = _mm_and_ps(shuffle<1, 3, 1, 1>(c0, c1).v, msk);
		r.col2.m_pk.v = _mm_and_ps(shuffle<2, 3, 0, 0>(c1, c2).v, msk);

		return detv;
	}


	inline sse_f64pk adjoint(const smat_core<f64, 3, 3>& a, smat_core<f64, 3, 3>& r)
	{
		// calculate co-factors

		// row 0

		sse_f64pk a0 = shuffle<1, 0>(a.col0.m_pk0, a.col0.m_pk1);
		sse_f64pk a1 = shuffle<1, 0>(a.col1.m_pk0, a.col1.m_pk1);
		sse_f64pk b1 = shuffle<0, 1>(a.col1.m_pk1, a.col1.m_pk0);
		sse_f64pk b2 = shuffle<0, 1>(a.col2.m_pk1, a.col2.m_pk0);

		sse_f64pk c00 = a1 * b2;
		sse_f64pk c01 = a0 * b2;
		sse_f64pk c02 = a0 * b1;

		r.col0.m_pk0 = shuffle<0,1>(c00, c01) - shuffle<1,0>(c00, c01);
		r.col0.m_pk1 = c02 - c02.dup_high();

		// row 1

		a0 = shuffle<0, 0>(a.col0.m_pk0, a.col0.m_pk1);
		a1 = shuffle<0, 0>(a.col1.m_pk0, a.col1.m_pk1);
		b1 = shuffle<0, 0>(a.col1.m_pk1, a.col1.m_pk0);
		b2 = shuffle<0, 0>(a.col2.m_pk1, a.col2.m_pk0);

		sse_f64pk c10 = a1 * b2;
		sse_f64pk c11 = a0 * b2;
		sse_f64pk c12 = a0 * b1;

		r.col1.m_pk0 = shuffle<1,0>(c10, c11) - shuffle<0,1>(c10, c11);
		r.col1.m_pk1 = c12.dup_high() - c12;

		// row 3

		a0 = shuffle<0, 1>(a.col0.m_pk0, a.col0.m_pk0);
		a1 = shuffle<0, 1>(a.col1.m_pk0, a.col1.m_pk0);
		b1 = shuffle<1, 0>(a.col1.m_pk0, a.col1.m_pk0);
		b2 = shuffle<1, 0>(a.col2.m_pk0, a.col2.m_pk0);

		sse_f64pk c20 = a1 * b2;
		sse_f64pk c21 = a0 * b2;
		sse_f64pk c22 = a0 * b1;

		r.col2.m_pk0 = shuffle<0,1>(c20, c21) - shuffle<1,0>(c20, c21);
		r.col2.m_pk1 = c22 - c22.dup_high();

		// calculate determinant

		sse_f64pk detv = mul_s( r.col0.m_pk0, a.col0.m_pk0 );
		detv = add_s(detv, mul_s(r.col1.m_pk0, a.col0.m_pk0.dup_high()));
		detv = add_s(detv, mul_s(r.col2.m_pk0, a.col0.m_pk1));

		return detv;
	}



	/******************************************************
	 *
	 *  Matrix 4 x 4
	 *
	 *  Let X = [A B; C D], then
	 *
	 *  |X| = |A| |D| + |B| |C| - tr(A# * B * D# * C)
	 *  Here, A# and D# are adjoints of A and D
	 *
	 ******************************************************/

	inline f32 det(const smat_core<f32, 4, 4>& X)
	{
		// pre-determinant for 2x2 blocks

		smat_core<f32,2,2> A = merge_low (X.col0.m_pk, X.col1.m_pk);
		smat_core<f32,2,2> C = merge_high(X.col0.m_pk, X.col1.m_pk);

		sse_f32pk dA = pre_det(A);
		sse_f32pk dC = pre_det(C);

		smat_core<f32,2,2> B = merge_low (X.col2.m_pk, X.col3.m_pk);
		smat_core<f32,2,2> D = merge_high(X.col2.m_pk, X.col3.m_pk);

		sse_f32pk dB = pre_det(B);
		sse_f32pk dD = pre_det(D);

		// combine terms

		sse_f32pk u1 = merge_low( dA, dB );
		sse_f32pk u2 = merge_low( dD, dC );
		sse_f32pk comb = shuffle<0,2,0,2>(u1, u2) - shuffle<1,3,1,3>(u1, u2);
		comb = comb * comb.dup_high();
		comb = add_s(comb, comb.dup2_high());

		// adjoint matrices for a and d

		smat_core<f32,2,2> adj_mask = adjoint_signmask_f32();

		smat_core<f32,2,2> Aa = adjoint_mat(A, adj_mask);
		smat_core<f32,2,2> Da = adjoint_mat(D, adj_mask);

		// Q = A# * B * D# * C

		smat_core<f32,2,2> AaB = mm2x2(Aa, B);
		smat_core<f32,2,2> DaC = mm2x2(Da, C);
		sse_f32pk qtr = mm2x2_trace_p(AaB, DaC);

		return sub_s(comb, qtr).to_scalar();
	}


	inline f64 det(const smat_core<f64, 4, 4>& X)
	{
		// pre-determinant for 2 x 2 blocks

		smat_core<f64, 2, 2> A, B, C, D;

		A.col0.m_pk = X.col0.m_pk0;
		A.col1.m_pk = X.col1.m_pk0;
		C.col0.m_pk = X.col0.m_pk1;
		C.col1.m_pk = X.col1.m_pk1;

		sse_f64pk dA = pre_det(A);
		sse_f64pk dC = pre_det(C);

		B.col0.m_pk = X.col2.m_pk0;
		B.col1.m_pk = X.col3.m_pk0;
		D.col0.m_pk = X.col2.m_pk1;
		D.col1.m_pk = X.col3.m_pk1;

		sse_f64pk dB = pre_det(B);
		sse_f64pk dD = pre_det(D);

		// combine terms

		sse_f64pk ab_p = unpack_low (dA, dB);
		sse_f64pk ab_n = unpack_high(dA, dB);
		sse_f64pk dc_p = unpack_low (dD, dC);
		sse_f64pk dc_n = unpack_high(dD, dC);

		ab_p = ab_p - ab_n;
		dc_p = dc_p - dc_n;

		sse_f64pk comb = ab_p * dc_p;
		comb = add_s(comb, comb.dup_high());

		// adjoint matrices for a and d

		smat_core<f64,2,2> adj_mask = adjoint_signmask_f64();

		smat_core<f64,2,2> Aa = adjoint_mat(A, adj_mask);
		smat_core<f64,2,2> Da = adjoint_mat(D, adj_mask);

		// Q = A# * B * D# * C

		smat_core<f64,2,2> AaB = mm2x2(Aa, B);
		smat_core<f64,2,2> DaC = mm2x2(Da, C);
		sse_f64pk qtr = mm2x2_trace_p(AaB, DaC);

		return sub_s(comb, qtr).to_scalar();
	}


	inline sse_f32pk adjoint(const smat_core<f32,4,4>& X, smat_core<f32,4,4>& Y)
	{
		// blocking and evaluate pre-determinant

		smat_core<f32,2,2> A = merge_low (X.col0.m_pk, X.col1.m_pk);
		smat_core<f32,2,2> C = merge_high(X.col0.m_pk, X.col1.m_pk);

		sse_f32pk dA = hdiff(pre_det(A)).bsx<0>();
		sse_f32pk dC = hdiff(pre_det(C)).bsx<0>();

		smat_core<f32,2,2> B = merge_low (X.col2.m_pk, X.col3.m_pk);
		smat_core<f32,2,2> D = merge_high(X.col2.m_pk, X.col3.m_pk);

		sse_f32pk dB = hdiff(pre_det(B)).bsx<0>();
		sse_f32pk dD = hdiff(pre_det(D)).bsx<0>();

		// adjoint matrices

		smat_core<f32,2,2> adj_mask = adjoint_signmask_f32();

		smat_core<f32,2,2> Aa = adjoint_mat(A, adj_mask);
		smat_core<f32,2,2> Ba = adjoint_mat(B, adj_mask);
		smat_core<f32,2,2> Ca = adjoint_mat(C, adj_mask);
		smat_core<f32,2,2> Da = adjoint_mat(D, adj_mask);

		// incomplete partial inverses

		smat_core<f32,2,2> IA = mm2x2(B, mm2x2(Da, C));
		smat_core<f32,2,2> IB = mm2x2(D, mm2x2(Ba, A));
		smat_core<f32,2,2> IC = mm2x2(A, mm2x2(Ca, D));
		smat_core<f32,2,2> ID = mm2x2(C, mm2x2(Aa, B));

		// determinant

		sse_f32pk comb = add_s(mul_s(dA, dD), mul_s(dB, dC));
		sse_f32pk qtr = mm2x2_trace_p(Aa, IA);
		sse_f32pk detv = sub_s(comb, qtr);

		// complete partial inverses

		A *= dD;
		B *= dC;
		C *= dB;
		D *= dA;

		IA = adjoint_mat(A - IA, adj_mask);
		IB = adjoint_mat(C - IB, adj_mask);
		IC = adjoint_mat(B - IC, adj_mask);
		ID = adjoint_mat(D - ID, adj_mask);

		// assemble into the inverse matrix

		Y.col0.m_pk = merge_low (IA.col01_pk, IC.col01_pk);
		Y.col1.m_pk = merge_high(IA.col01_pk, IC.col01_pk);
		Y.col2.m_pk = merge_low (IB.col01_pk, ID.col01_pk);
		Y.col3.m_pk = merge_high(IB.col01_pk, ID.col01_pk);

		return detv;
	}


	inline sse_f64pk adjoint(const smat_core<f64,4,4>& X, smat_core<f64,4,4>& Y)
	{
		// blocking and evaluate pre-determinant

		smat_core<f64, 2, 2> A, B, C, D;

		A.col0.m_pk = X.col0.m_pk0;
		A.col1.m_pk = X.col1.m_pk0;
		C.col0.m_pk = X.col0.m_pk1;
		C.col1.m_pk = X.col1.m_pk1;

		sse_f64pk dA = hdiff(pre_det(A)).bsx<0>();
		sse_f64pk dC = hdiff(pre_det(C)).bsx<0>();

		B.col0.m_pk = X.col2.m_pk0;
		B.col1.m_pk = X.col3.m_pk0;
		D.col0.m_pk = X.col2.m_pk1;
		D.col1.m_pk = X.col3.m_pk1;

		sse_f64pk dB = hdiff(pre_det(B)).bsx<0>();
		sse_f64pk dD = hdiff(pre_det(D)).bsx<0>();

		// adjoint matrices

		smat_core<f64,2,2> adj_mask = adjoint_signmask_f64();

		smat_core<f64,2,2> Aa = adjoint_mat(A, adj_mask);
		smat_core<f64,2,2> Ba = adjoint_mat(B, adj_mask);
		smat_core<f64,2,2> Ca = adjoint_mat(C, adj_mask);
		smat_core<f64,2,2> Da = adjoint_mat(D, adj_mask);

		// incomplete partial inverses

		smat_core<f64,2,2> IA = mm2x2(B, mm2x2(Da, C));
		smat_core<f64,2,2> IB = mm2x2(D, mm2x2(Ba, A));
		smat_core<f64,2,2> IC = mm2x2(A, mm2x2(Ca, D));
		smat_core<f64,2,2> ID = mm2x2(C, mm2x2(Aa, B));

		// determinant

		sse_f64pk comb = add_s(mul_s(dA, dD), mul_s(dB, dC));
		sse_f64pk qtr = mm2x2_trace_p(Aa, IA);
		sse_f64pk detv = sub_s(comb, qtr);

		// complete partial inverses

		A *= dD;
		B *= dC;
		C *= dB;
		D *= dA;

		IA = adjoint_mat(A - IA, adj_mask);
		IB = adjoint_mat(C - IB, adj_mask);
		IC = adjoint_mat(B - IC, adj_mask);
		ID = adjoint_mat(D - ID, adj_mask);

		// assemble into the inverse matrix

		Y.col0.m_pk0 = IA.col0.m_pk;
		Y.col0.m_pk1 = IC.col0.m_pk;
		Y.col1.m_pk0 = IA.col1.m_pk;
		Y.col1.m_pk1 = IC.col1.m_pk;
		Y.col2.m_pk0 = IB.col0.m_pk;
		Y.col2.m_pk1 = ID.col0.m_pk;
		Y.col3.m_pk0 = IB.col1.m_pk;
		Y.col3.m_pk1 = ID.col1.m_pk;

		return detv;
	}


	/**********************************
	 *
	 *  Generic inverse function
	 *
	 **********************************/

	template<typename T, int N>
	inline T inv(const smat_core<T, N, N>& A, smat_core<T, N, N>& R)
	{
		sse_pack<T> detv = adjoint(A, R);

		sse_pack<T> rdetv = rcp_s(detv).template bsx<0>();
		R *= rdetv;

		return detv.to_scalar();
	}


	/*******************************************
	 *
	 *  Generic solving function (for 2x2)
	 *
	 *******************************************/

	template<typename T, int N>
	inline void solve(const smat_core<T, N, N>& A, const sse_vec<T, N>& b, sse_vec<T, N>& x)
	{
		smat_core<T, N, N> R;
		sse_pack<T> detv = adjoint(A, R);

		sse_pack<T> rdetv = rcp_s(detv).template bsx<0>();
		sse_vec<T, N> scaled_b = b * rdetv;

		x = transform(R, scaled_b);
	}

	template<typename T, int N, int N2>
	inline void solve(const smat_core<T, N, N>& A, const smat_core<T, N, N2>& B, smat_core<T, N, N2>& X)
	{
		smat_core<T, N, N> R;
		sse_pack<T> detv = adjoint(A, R);

		sse_pack<T> rdetv = rcp_s(detv).template bsx<0>();

		if (N2 < N)
		{
			smat_core<T, N, N2> scaled_B = B * rdetv;
			mtimes_op<T, N, N, N2>::run(R, scaled_B, X);
		}
		else
		{
			R *= rdetv;
			mtimes_op<T, N, N, N2>::run(R, B, X);
		}
	}


	/*******************************************
	 *
	 *  Specialized solving function (for 2x2)
	 *
	 *******************************************/

	LSIMD_ENSURE_INLINE
	inline void solve(const smat_core<f32, 2, 2>& A, const sse_vec<f32, 2>& b, sse_vec<f32, 2>& x)
	{
		sse_f32pk ac = A.col01_pk.swizzle<3,0,2,1>();

		sse_f32pk px = b.m_pk.swizzle<0,1,1,0>() * ac;
		px = px - px.dup_high();

		sse_f32pk detv = hdiff(pre_det(A));

		__m128 msk = low2_mask_f32();
		sse_f32pk rdetv = rcp_s(detv).bsx<0>();
		rdetv.v = _mm_and_ps(rdetv.v, msk);

		x.m_pk = px * rdetv;
	}

	LSIMD_ENSURE_INLINE
	inline void solve(const smat_core<f32, 2, 2>& A, const smat_core<f32, 2, 2>& B, smat_core<f32, 2, 2>& X)
	{
		sse_f32pk ac = A.col01_pk.swizzle<3,0,2,1>();

		sse_f32pk px0 = B.col01_pk.swizzle<0,1,1,0>() * ac;
		sse_f32pk px1 = B.col01_pk.swizzle<2,3,3,2>() * ac;

		px0 = px0 - px0.dup_high();
		px1 = px1 - px1.dup_high();

		sse_f32pk detv = hdiff(pre_det(A));
		sse_f32pk rdetv = rcp_s(detv).bsx<0>();

		X.col01_pk = merge_low(px0, px1) * rdetv;
	}

	LSIMD_ENSURE_INLINE
	inline void solve(const smat_core<f32, 2, 2>& A, const smat_core<f32, 2, 3>& B, smat_core<f32, 2, 3>& X)
	{
		sse_f32pk ac = A.col01_pk.swizzle<3,0,2,1>();

		sse_f32pk px0 = B.col01_pk.swizzle<0,1,1,0>() * ac;
		sse_f32pk px1 = B.col01_pk.swizzle<2,3,3,2>() * ac;
		sse_f32pk px2 = B.col2z_pk.swizzle<0,1,1,0>() * ac;

		px0 = px0 - px0.dup_high();
		px1 = px1 - px1.dup_high();
		px2 = px2 - px2.dup_high();

		__m128 msk = low2_mask_f32();
		sse_f32pk detv = hdiff(pre_det(A));
		sse_f32pk rdetv = rcp_s(detv).bsx<0>();

		X.col01_pk = merge_low(px0, px1) * rdetv;

		rdetv.v = _mm_and_ps(rdetv.v, msk);
		X.col2z_pk = px2 * rdetv;
	}

	LSIMD_ENSURE_INLINE
	inline void solve(const smat_core<f32, 2, 2>& A, const smat_core<f32, 2, 4>& B, smat_core<f32, 2, 4>& X)
	{
		sse_f32pk ac = A.col01_pk.swizzle<3,0,2,1>();

		sse_f32pk px0 = B.col01_pk.swizzle<0,1,1,0>() * ac;
		sse_f32pk px1 = B.col01_pk.swizzle<2,3,3,2>() * ac;
		sse_f32pk px2 = B.col23_pk.swizzle<0,1,1,0>() * ac;
		sse_f32pk px3 = B.col23_pk.swizzle<2,3,3,2>() * ac;

		px0 = px0 - px0.dup_high();
		px1 = px1 - px1.dup_high();
		px2 = px2 - px2.dup_high();
		px3 = px3 - px3.dup_high();

		sse_f32pk detv = hdiff(pre_det(A));
		sse_f32pk rdetv = rcp_s(detv).bsx<0>();

		X.col01_pk = merge_low(px0, px1) * rdetv;
		X.col23_pk = merge_low(px2, px3) * rdetv;
	}

	LSIMD_ENSURE_INLINE
	inline void solve(const smat_core<f64, 2, 2>& A, const sse_vec<f64, 2>& b, sse_vec<f64, 2>& x)
	{
		sse_f64pk ac_p = shuffle<1,0>(A.col1.m_pk, A.col0.m_pk);
		sse_f64pk ac_n = shuffle<1,0>(A.col0.m_pk, A.col1.m_pk);

		sse_f64pk detv = hdiff(pre_det(A));
		sse_f64pk rdetv = rcp_s(detv).bsx<0>();

		sse_f64pk x_p = ac_p * b.m_pk;
		sse_f64pk x_n = (ac_n * b.m_pk).swizzle<1,0>();

		x.m_pk = (x_p - x_n) * rdetv;
	}

	LSIMD_ENSURE_INLINE
	inline void solve(const smat_core<f64, 2, 2>& A, const smat_core<f64, 2, 2>& B, smat_core<f64, 2, 2>& X)
	{
		sse_f64pk ac_p = shuffle<1,0>(A.col1.m_pk, A.col0.m_pk);
		sse_f64pk ac_n = shuffle<1,0>(A.col0.m_pk, A.col1.m_pk);

		sse_f64pk x0_p = ac_p * B.col0.m_pk;
		sse_f64pk x0_n = (ac_n * B.col0.m_pk).swizzle<1,0>();

		sse_f64pk x1_p = ac_p * B.col1.m_pk;
		sse_f64pk x1_n = (ac_n * B.col1.m_pk).swizzle<1,0>();

		sse_f64pk detv = hdiff(pre_det(A));
		sse_f64pk rdetv = rcp_s(detv).bsx<0>();

		X.col0.m_pk = (x0_p - x0_n) * rdetv;
		X.col1.m_pk = (x1_p - x1_n) * rdetv;
	}

	LSIMD_ENSURE_INLINE
	inline void solve(const smat_core<f64, 2, 2>& A, const smat_core<f64, 2, 3>& B, smat_core<f64, 2, 3>& X)
	{
		sse_f64pk ac_p = shuffle<1,0>(A.col1.m_pk, A.col0.m_pk);
		sse_f64pk ac_n = shuffle<1,0>(A.col0.m_pk, A.col1.m_pk);

		sse_f64pk detv = hdiff(pre_det(A));
		sse_f64pk rdetv = rcp_s(detv).bsx<0>();

		sse_f64pk x0_p = ac_p * B.col0.m_pk;
		sse_f64pk x0_n = (ac_n * B.col0.m_pk).swizzle<1,0>();

		sse_f64pk x1_p = ac_p * B.col1.m_pk;
		sse_f64pk x1_n = (ac_n * B.col1.m_pk).swizzle<1,0>();

		X.col0.m_pk = (x0_p - x0_n) * rdetv;
		X.col1.m_pk = (x1_p - x1_n) * rdetv;

		sse_f64pk x2_p = ac_p * B.col2.m_pk;
		sse_f64pk x2_n = (ac_n * B.col2.m_pk).swizzle<1,0>();

		X.col2.m_pk = (x2_p - x2_n) * rdetv;
	}

	LSIMD_ENSURE_INLINE
	inline void solve(const smat_core<f64, 2, 2>& A, const smat_core<f64, 2, 4>& B, smat_core<f64, 2, 4>& X)
	{
		sse_f64pk ac_p = shuffle<1,0>(A.col1.m_pk, A.col0.m_pk);
		sse_f64pk ac_n = shuffle<1,0>(A.col0.m_pk, A.col1.m_pk);

		sse_f64pk detv = hdiff(pre_det(A));
		sse_f64pk rdetv = rcp_s(detv).bsx<0>();

		sse_f64pk x0_p = ac_p * B.col0.m_pk;
		sse_f64pk x0_n = (ac_n * B.col0.m_pk).swizzle<1,0>();

		sse_f64pk x1_p = ac_p * B.col1.m_pk;
		sse_f64pk x1_n = (ac_n * B.col1.m_pk).swizzle<1,0>();

		X.col0.m_pk = (x0_p - x0_n) * rdetv;
		X.col1.m_pk = (x1_p - x1_n) * rdetv;

		sse_f64pk x2_p = ac_p * B.col2.m_pk;
		sse_f64pk x2_n = (ac_n * B.col2.m_pk).swizzle<1,0>();

		sse_f64pk x3_p = ac_p * B.col3.m_pk;
		sse_f64pk x3_n = (ac_n * B.col3.m_pk).swizzle<1,0>();

		X.col2.m_pk = (x2_p - x2_n) * rdetv;
		X.col3.m_pk = (x3_p - x3_n) * rdetv;
	}

} }

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif 
