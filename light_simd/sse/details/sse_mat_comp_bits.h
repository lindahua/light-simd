/**
 * @file sse_mat_comp_bits.h
 *
 * Internal implementation for SSE Matrix computation
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_MAT_COMP_BITS_H_
#define LSIMD_SSE_MAT_COMP_BITS_H_

#include "sse_mat_bits.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4141)
#endif


namespace lsimd { namespace sse {

	/********************************************
	 *
	 *  Evaluation for M x 2
	 *
	 ********************************************/

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline smat_core<T,M,2> operator + (const smat_core<T,M,2>& a, const smat_core<T,M,2>& b)
	{
		smat_core<T,M,2> r;
		r.col0 = a.col0 + b.col0;
		r.col1 = a.col1 + b.col1;
		return r;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline smat_core<T,M,2> operator - (const smat_core<T,M,2>& a, const smat_core<T,M,2>& b)
	{
		smat_core<T,M,2> r;
		r.col0 = a.col0 - b.col0;
		r.col1 = a.col1 - b.col1;
		return r;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline smat_core<T,M,2> operator % (const smat_core<T,M,2>& a, const smat_core<T,M,2>& b)
	{
		smat_core<T,M,2> r;
		r.col0 = a.col0 % b.col0;
		r.col1 = a.col1 % b.col1;
		return r;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline smat_core<T,M,2> operator * (const smat_core<T,M,2>& a, const sse_pack<T> s)
	{
		smat_core<T,M,2> r;
		r.col0 = a.col0 * s;
		r.col1 = a.col1 * s;
		return r;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline void operator += (smat_core<T,M,2>& a, const smat_core<T,M,2>& b)
	{
		a.col0 += b.col0;
		a.col1 += b.col1;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline void operator -= (smat_core<T,M,2>& a, const smat_core<T,M,2>& b)
	{
		a.col0 -= b.col0;
		a.col1 -= b.col1;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline void operator %= (smat_core<T,M,2>& a, const smat_core<T,M,2>& b)
	{
		a.col0 %= b.col0;
		a.col1 %= b.col1;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline void operator *= (smat_core<T,M,2>& a, const sse_pack<T> s)
	{
		a.col0 *= s;
		a.col1 *= s;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline sse_vec<T,M> transform(const smat_core<T,M,2>& a, sse_vec<T,2> x)
	{
		sse_vec<T,M> y =
				a.col0 * x.template bsx_pk<0>() +
				a.col1 * x.template bsx_pk<1>();
		return y;
	}


	/********************************************
	 *
	 *  Evaluation for M x 3
	 *
	 ********************************************/

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline smat_core<T,M,3> operator + (const smat_core<T,M,3>& a, const smat_core<T,M,3>& b)
	{
		smat_core<T,M,3> r;
		r.col0 = a.col0 + b.col0;
		r.col1 = a.col1 + b.col1;
		r.col2 = a.col2 + b.col2;
		return r;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline smat_core<T,M,3> operator - (const smat_core<T,M,3>& a, const smat_core<T,M,3>& b)
	{
		smat_core<T,M,3> r;
		r.col0 = a.col0 - b.col0;
		r.col1 = a.col1 - b.col1;
		r.col2 = a.col2 - b.col2;
		return r;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline smat_core<T,M,3> operator % (const smat_core<T,M,3>& a, const smat_core<T,M,3>& b)
	{
		smat_core<T,M,3> r;
		r.col0 = a.col0 % b.col0;
		r.col1 = a.col1 % b.col1;
		r.col2 = a.col2 % b.col2;
		return r;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline smat_core<T,M,3> operator * (const smat_core<T,M,3>& a, const sse_pack<T> s)
	{
		smat_core<T,M,3> r;
		r.col0 = a.col0 * s;
		r.col1 = a.col1 * s;
		r.col2 = a.col2 * s;
		return r;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline void operator += (smat_core<T,M,3>& a, const smat_core<T,M,3>& b)
	{
		a.col0 += b.col0;
		a.col1 += b.col1;
		a.col2 += b.col2;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline void operator -= (smat_core<T,M,3>& a, const smat_core<T,M,3>& b)
	{
		a.col0 -= b.col0;
		a.col1 -= b.col1;
		a.col2 -= b.col2;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline void operator %= (smat_core<T,M,3>& a, const smat_core<T,M,3>& b)
	{
		a.col0 %= b.col0;
		a.col1 %= b.col1;
		a.col2 %= b.col2;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline void operator *= (smat_core<T,M,3>& a, const sse_pack<T> s)
	{
		a.col0 *= s;
		a.col1 *= s;
		a.col2 *= s;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline sse_vec<T,M> transform(const smat_core<T,M,3>& a, sse_vec<T,3> x)
	{
		sse_vec<T,M> y =
				a.col0 * x.template bsx_pk<0>() +
				a.col1 * x.template bsx_pk<1>() +
				a.col2 * x.template bsx_pk<2>();
		return y;
	}


	/********************************************
	 *
	 *  Evaluation for M x 4
	 *
	 ********************************************/

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline smat_core<T,M,4> operator + (const smat_core<T,M,4>& a, const smat_core<T,M,4>& b)
	{
		smat_core<T,M,4> r;
		r.col0 = a.col0 + b.col0;
		r.col1 = a.col1 + b.col1;
		r.col2 = a.col2 + b.col2;
		r.col3 = a.col3 + b.col3;
		return r;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline smat_core<T,M,4> operator - (const smat_core<T,M,4>& a, const smat_core<T,M,4>& b)
	{
		smat_core<T,M,4> r;
		r.col0 = a.col0 - b.col0;
		r.col1 = a.col1 - b.col1;
		r.col2 = a.col2 - b.col2;
		r.col3 = a.col3 - b.col3;
		return r;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline smat_core<T,M,4> operator % (const smat_core<T,M,4>& a, const smat_core<T,M,4>& b)
	{
		smat_core<T,M,4> r;
		r.col0 = a.col0 % b.col0;
		r.col1 = a.col1 % b.col1;
		r.col2 = a.col2 % b.col2;
		r.col3 = a.col3 % b.col3;
		return r;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline smat_core<T,M,4> operator * (const smat_core<T,M,4>& a, const sse_pack<T> s)
	{
		smat_core<T,M,4> r;
		r.col0 = a.col0 * s;
		r.col1 = a.col1 * s;
		r.col2 = a.col2 * s;
		r.col3 = a.col3 * s;
		return r;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline void operator += (smat_core<T,M,4>& a, const smat_core<T,M,4>& b)
	{
		a.col0 += b.col0;
		a.col1 += b.col1;
		a.col2 += b.col2;
		a.col3 += b.col3;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline void operator -= (smat_core<T,M,4>& a, const smat_core<T,M,4>& b)
	{
		a.col0 -= b.col0;
		a.col1 -= b.col1;
		a.col2 -= b.col2;
		a.col3 -= b.col3;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline void operator %= (smat_core<T,M,4>& a, const smat_core<T,M,4>& b)
	{
		a.col0 %= b.col0;
		a.col1 %= b.col1;
		a.col2 %= b.col2;
		a.col3 %= b.col3;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline void operator *= (smat_core<T,M,4>& a, const sse_pack<T> s)
	{
		a.col0 *= s;
		a.col1 *= s;
		a.col2 *= s;
		a.col3 *= s;
	}

	template<typename T, int M>
	LSIMD_ENSURE_INLINE
	inline sse_vec<T,M> transform(const smat_core<T,M,4>& a, sse_vec<T,4> x)
	{
		sse_vec<T,M> y0 =
				a.col0 * x.template bsx_pk<0>() +
				a.col1 * x.template bsx_pk<1>();

		sse_vec<T,M> y1 =
				a.col2 * x.template bsx_pk<2>() +
				a.col3 * x.template bsx_pk<3>();

		return y0 + y1;
	}


	/********************************************
	 *
	 *  Specialized Evaluation for f32 2 x 2
	 *
	 ********************************************/

	LSIMD_ENSURE_INLINE
	inline smat_core<f32,2,2> operator + (const smat_core<f32,2,2>& a, const smat_core<f32,2,2>& b)
	{
		return a.col01_pk + b.col01_pk;
	}

	LSIMD_ENSURE_INLINE
	inline smat_core<f32,2,2> operator - (const smat_core<f32,2,2>& a, const smat_core<f32,2,2>& b)
	{
		return a.col01_pk - b.col01_pk;
	}

	LSIMD_ENSURE_INLINE
	inline smat_core<f32,2,2> operator % (const smat_core<f32,2,2>& a, const smat_core<f32,2,2>& b)
	{
		return a.col01_pk * b.col01_pk;
	}

	LSIMD_ENSURE_INLINE
	inline smat_core<f32,2,2> operator * (const smat_core<f32,2,2>& a, const sse_f32pk& s)
	{
		return a.col01_pk * s;
	}

	LSIMD_ENSURE_INLINE
	inline void operator += (smat_core<f32,2,2>& a, const smat_core<f32,2,2>& b)
	{
		a.col01_pk = a.col01_pk + b.col01_pk;
	}

	LSIMD_ENSURE_INLINE
	inline void operator -= (smat_core<f32,2,2>& a, const smat_core<f32,2,2>& b)
	{
		a.col01_pk = a.col01_pk - b.col01_pk;
	}

	LSIMD_ENSURE_INLINE
	inline void operator %= (smat_core<f32,2,2>& a, const smat_core<f32,2,2>& b)
	{
		a.col01_pk = a.col01_pk * b.col01_pk;
	}

	LSIMD_ENSURE_INLINE
	inline void operator *= (smat_core<f32,2,2>& a, const sse_f32pk& s)
	{
		a.col01_pk = a.col01_pk * s;
	}

	LSIMD_ENSURE_INLINE
	inline sse_vec<f32,2> transform(const smat_core<f32,2,2>& a, const sse_vec<f32,2>& x)
	{
		sse_f32pk p = a.col01_pk * x.m_pk.swizzle<0, 0, 1, 1>();
		p = p + p.dup_low();
		return sse_vec<f32,2>(p.shift_front<2>());
	}


	/********************************************
	 *
	 *  Specialized Evaluation for f32 2 x 3
	 *
	 ********************************************/

	LSIMD_ENSURE_INLINE
	inline smat_core<f32,2,3> operator + (const smat_core<f32,2,3>& a, const smat_core<f32,2,3>& b)
	{
		smat_core<f32,2,3> r;
		r.col01_pk = a.col01_pk + b.col01_pk;
		r.col2z_pk = a.col2z_pk + b.col2z_pk;
		return r;
	}

	LSIMD_ENSURE_INLINE
	inline smat_core<f32,2,3> operator - (const smat_core<f32,2,3>& a, const smat_core<f32,2,3>& b)
	{
		smat_core<f32,2,3> r;
		r.col01_pk = a.col01_pk - b.col01_pk;
		r.col2z_pk = a.col2z_pk - b.col2z_pk;
		return r;
	}

	LSIMD_ENSURE_INLINE
	inline smat_core<f32,2,3> operator % (const smat_core<f32,2,3>& a, const smat_core<f32,2,3>& b)
	{
		smat_core<f32,2,3> r;
		r.col01_pk = a.col01_pk * b.col01_pk;
		r.col2z_pk = a.col2z_pk * b.col2z_pk;
		return r;
	}

	LSIMD_ENSURE_INLINE
	inline smat_core<f32,2,3> operator * (const smat_core<f32,2,3>& a, const sse_f32pk& s)
	{
		smat_core<f32,2,3> r;
		r.col01_pk = a.col01_pk * s;
		r.col2z_pk = a.col2z_pk * s;
		return r;
	}

	LSIMD_ENSURE_INLINE
	inline void operator += (smat_core<f32,2,3>& a, const smat_core<f32,2,3>& b)
	{
		a.col01_pk = a.col01_pk + b.col01_pk;
		a.col2z_pk = a.col2z_pk + b.col2z_pk;
	}

	LSIMD_ENSURE_INLINE
	inline void operator -= (smat_core<f32,2,3>& a, const smat_core<f32,2,3>& b)
	{
		a.col01_pk = a.col01_pk - b.col01_pk;
		a.col2z_pk = a.col2z_pk - b.col2z_pk;
	}

	LSIMD_ENSURE_INLINE
	inline void operator %= (smat_core<f32,2,3>& a, const smat_core<f32,2,3>& b)
	{
		a.col01_pk = a.col01_pk * b.col01_pk;
		a.col2z_pk = a.col2z_pk * b.col2z_pk;
	}

	LSIMD_ENSURE_INLINE
	inline void operator *= (smat_core<f32,2,3>& a, const sse_f32pk& s)
	{
		a.col01_pk = a.col01_pk * s;
		a.col2z_pk = a.col2z_pk * s;
	}

	LSIMD_ENSURE_INLINE
	inline sse_vec<f32,2> transform(const smat_core<f32,2,3>& a, const sse_vec<f32,3>& x)
	{
		sse_f32pk p1 = unpack_low(x.m_pk, x.m_pk) * a.col01_pk;
		sse_f32pk p2 = unpack_high(x.m_pk, x.m_pk) * a.col2z_pk;

		p1 = p1 + p2;
		p1 = p1 + p1.dup_low();

		return sse_vec<f32, 2>(p1.shift_front<2>());
	}


	/********************************************
	 *
	 *  Specialized Evaluation for f32 2 x 4
	 *
	 ********************************************/

	LSIMD_ENSURE_INLINE
	inline smat_core<f32,2,4> operator + (const smat_core<f32,2,4>& a, const smat_core<f32,2,4>& b)
	{
		smat_core<f32,2,4> r;
		r.col01_pk = a.col01_pk + b.col01_pk;
		r.col23_pk = a.col23_pk + b.col23_pk;
		return r;
	}

	LSIMD_ENSURE_INLINE
	inline smat_core<f32,2,4> operator - (const smat_core<f32,2,4>& a, const smat_core<f32,2,4>& b)
	{
		smat_core<f32,2,4> r;
		r.col01_pk = a.col01_pk - b.col01_pk;
		r.col23_pk = a.col23_pk - b.col23_pk;
		return r;
	}

	LSIMD_ENSURE_INLINE
	inline smat_core<f32,2,4> operator % (const smat_core<f32,2,4>& a, const smat_core<f32,2,4>& b)
	{
		smat_core<f32,2,4> r;
		r.col01_pk = a.col01_pk * b.col01_pk;
		r.col23_pk = a.col23_pk * b.col23_pk;
		return r;
	}

	LSIMD_ENSURE_INLINE
	inline smat_core<f32,2,4> operator * (const smat_core<f32,2,4>& a, const sse_f32pk& s)
	{
		smat_core<f32,2,4> r;
		r.col01_pk = a.col01_pk * s;
		r.col23_pk = a.col23_pk * s;
		return r;
	}

	LSIMD_ENSURE_INLINE
	inline void operator += (smat_core<f32,2,4>& a, const smat_core<f32,2,4>& b)
	{
		a.col01_pk = a.col01_pk + b.col01_pk;
		a.col23_pk = a.col23_pk + b.col23_pk;
	}

	LSIMD_ENSURE_INLINE
	inline void operator -= (smat_core<f32,2,4>& a, const smat_core<f32,2,4>& b)
	{
		a.col01_pk = a.col01_pk - b.col01_pk;
		a.col23_pk = a.col23_pk - b.col23_pk;
	}

	LSIMD_ENSURE_INLINE
	inline void operator %= (smat_core<f32,2,4>& a, const smat_core<f32,2,4>& b)
	{
		a.col01_pk = a.col01_pk * b.col01_pk;
		a.col23_pk = a.col23_pk * b.col23_pk;
	}

	LSIMD_ENSURE_INLINE
	inline void operator *= (smat_core<f32,2,4>& a, const sse_f32pk& s)
	{
		a.col01_pk = a.col01_pk * s;
		a.col23_pk = a.col23_pk * s;
	}

	LSIMD_ENSURE_INLINE
	inline sse_vec<f32,2> transform(const smat_core<f32,2,4>& a, const sse_vec<f32,4>& x)
	{
		sse_f32pk p1 = unpack_low(x.m_pk, x.m_pk) * a.col01_pk;
		sse_f32pk p2 = unpack_high(x.m_pk, x.m_pk) * a.col23_pk;

		p1 = p1 + p2;
		p1 = p1 + p1.dup_low();
		return sse_vec<f32, 2>(p1.shift_front<2>());
	}


} }

#ifdef _MSC_VER
#pragma warning(pop)
#endif


#endif /* SSE_MAT_COMP_BITS_H_ */
