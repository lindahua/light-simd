/**
 * @file sse_mat_bits.h
 *
 * Internal implementation for SSE-based fixed-size matrix
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_MAT_BITS_H_
#define LSIMD_SSE_MAT_BITS_H_

#include "../sse_vec.h"

namespace lsimd { namespace sse {

	template<typename T, int M, int N> struct smat_core;


	/********************************************
	 *
	 *  specialized core classes for f32 2 x N
	 *
	 ********************************************/


	template<>
	struct smat_core<f32, 2, 2>
	{
		typedef sse_vec<f32, 2> vec_t;

		sse_f32pk col01_pk;

		LSIMD_ENSURE_INLINE
		smat_core() { }

		LSIMD_ENSURE_INLINE
		smat_core( zero_t ) : col01_pk( zero_t() ) { }

		LSIMD_ENSURE_INLINE
		smat_core(const sse_f32pk& c01) : col01_pk( c01 ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, AlignT)
		{
			col01_pk.load(x, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, int ldim, AlignT)
		{
			sse_f32pk p0, p1;

			p0.partial_load<2>(x);
			p1.partial_load<2>(x + ldim);

			col01_pk = merge_low(p0, p1);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, AlignT)
		{
			sse_f32pk t(x, AlignT());
			col01_pk = t.swizzle<0,2,1,3>();
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, int ldim, AlignT)
		{
			sse_f32pk p0, p1;

			p0.partial_load<2>(x);
			p1.partial_load<2>(x + ldim);

			col01_pk = unpack_low(p0, p1);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, AlignT) const
		{
			col01_pk.store(x, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, int ldim, AlignT) const
		{
			sse::partial_store<2>(x, col01_pk.v);
			sse::partial_store<2>(x + ldim, col01_pk.dup_high().v);
		}

		LSIMD_ENSURE_INLINE f32 trace() const
		{
			return col01_pk.e[0] + col01_pk.e[3];
		}


		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return col01_pk.test_equal(r);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [2 x 2]:\n");

			std::printf("    col01 = ");
			col01_pk.dump(fmt);
			std::printf("\n");
		}
	};


	template<>
	struct smat_core<f32, 2, 3>
	{
		typedef sse_vec<f32, 2> vec_t;

		sse_f32pk col01_pk;
		sse_f32pk col2z_pk;

		LSIMD_ENSURE_INLINE
		smat_core() { }

		LSIMD_ENSURE_INLINE
		smat_core( zero_t )
		: col01_pk( zero_t() ), col2z_pk( zero_t() ) { }

		LSIMD_ENSURE_INLINE
		smat_core(const sse_f32pk& c01, const sse_f32pk& c2)
		: col01_pk( c01 ), col2z_pk( c2 ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, AlignT)
		{
			col01_pk.load(x, AlignT());
			col2z_pk.partial_load<2>(x + 4);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, int ldim, AlignT)
		{
			sse_f32pk p0, p1;

			p0.partial_load<2>(x);
			p1.partial_load<2>(x + ldim);

			col01_pk = merge_low(p0, p1);
			col2z_pk.partial_load<2>(x + 2 * ldim);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, AlignT)
		{
			_load_trans(x, x + 3);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, int ldim, AlignT)
		{
			_load_trans(x, x + ldim);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, AlignT) const
		{
			col01_pk.store(x, AlignT());
			col2z_pk.partial_store<2>(x + 4);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, int ldim, AlignT) const
		{
			col01_pk.partial_store<2>(x);
			col01_pk.dup_high().partial_store<2>(x + ldim);
			col2z_pk.partial_store<2>(x + 2 * ldim);
		}

		LSIMD_ENSURE_INLINE f32 trace() const
		{
			return col01_pk.e[0] + col01_pk.e[3];
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return col01_pk.test_equal(r) && col2z_pk.test_equal(r[4], r[5], 0.f, 0.f);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [2 x 3]:\n");

			std::printf("    col01 = ");
			col01_pk.dump(fmt);
			std::printf("\n");

			std::printf("    col2z = ");
			col2z_pk.dump(fmt);
			std::printf("\n");
		}

	private:
		LSIMD_ENSURE_INLINE void _load_trans(const f32 *r0, const f32 *r1)
		{
			sse_f32pk pr0a, pr0b, pr1a, pr1b;

			pr0a.partial_load<2>(r0);
			pr0b.partial_load<1>(r0 + 2);
			pr1a.partial_load<2>(r1);
			pr1b.partial_load<1>(r1 + 2);

			col01_pk = unpack_low(pr0a, pr1a);
			col2z_pk = unpack_low(pr0b, pr1b);
		}
	};



	template<>
	struct smat_core<f32, 2, 4>
	{
		typedef sse_vec<f32, 2> vec_t;

		sse_f32pk col01_pk;
		sse_f32pk col23_pk;

		LSIMD_ENSURE_INLINE
		smat_core() { }

		LSIMD_ENSURE_INLINE
		smat_core( zero_t )
		: col01_pk( zero_t() ), col23_pk( zero_t() ) { }

		LSIMD_ENSURE_INLINE
		smat_core(const sse_f32pk& c01, const sse_f32pk& c2)
		: col01_pk( c01 ), col23_pk( c2 ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, AlignT)
		{
			col01_pk.load(x, AlignT());
			col23_pk.load(x + 4, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, int ldim, AlignT)
		{
			const f32 *x2 = x + 2 * ldim;

			sse_f32pk p0, p1, p2, p3;

			p0.partial_load<2>(x);
			p1.partial_load<2>(x + ldim);
			p2.partial_load<2>(x2);
			p3.partial_load<2>(x2 + ldim);

			col01_pk = merge_low(p0, p1);
			col23_pk = merge_low(p2, p3);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, AlignT)
		{
			_load_trans(x, x + 4, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, int ldim, AlignT)
		{
			_load_trans(x, x + ldim, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, AlignT) const
		{
			col01_pk.store(x, AlignT());
			col23_pk.store(x + 4, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, int ldim, AlignT) const
		{
			f32 *x2 = x + 2 * ldim;

			col01_pk.partial_store<2>(x);
			col01_pk.dup_high().partial_store<2>(x + ldim);

			col23_pk.partial_store<2>(x2);
			col23_pk.dup_high().partial_store<2>(x2 + ldim);
		}

		LSIMD_ENSURE_INLINE f32 trace() const
		{
			return col01_pk.e[0] + col01_pk.e[3];
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return col01_pk.test_equal(r) && col23_pk.test_equal(r + 4);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [2 x 4]:\n");

			std::printf("    col01 = ");
			col01_pk.dump(fmt);
			std::printf("\n");

			std::printf("    col2z = ");
			col23_pk.dump(fmt);
			std::printf("\n");
		}

	private:
		template<typename AlignT>
		LSIMD_ENSURE_INLINE void _load_trans(const f32 *r0, const f32 *r1, AlignT)
		{
			sse_f32pk pr0(r0, AlignT());
			sse_f32pk pr1(r1, AlignT());

			col01_pk = unpack_low(pr0, pr1);
			col23_pk = unpack_high(pr0, pr1);
		}
	};




	/********************************************
	 *
	 *  specialized core classes for f32 3 x N
	 *
	 ********************************************/

	template<>
	struct smat_core<f32, 3, 2>
	{
		typedef sse_vec<f32, 3> vec_t;

		vec_t col0;
		vec_t col1;

		LSIMD_ENSURE_INLINE
		smat_core() { }

		LSIMD_ENSURE_INLINE
		smat_core( zero_t )
		: col0( zero_t() ), col1( zero_t() ){ }

		LSIMD_ENSURE_INLINE
		smat_core(const vec_t& v0, const vec_t& v1)
		: col0(v0), col1(v1) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + 3, unaligned_t());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, int ldim, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + ldim, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, AlignT)
		{
			sse_f32pk p01, p2;

			p01.load(x, AlignT());
			p2.partial_load<2>(x + 4);

			p01 = p01.swizzle<0,2,1,3>();
			p2 = p2.swizzle<0,2,1,3>();

			col0.m_pk = merge_low (p01, p2);
			col1.m_pk = merge_high(p01, p2);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, int ldim, AlignT)
		{
			sse_f32pk p0, p1, p2;

			p0.partial_load<2>(x);
			p1.partial_load<2>(x + ldim);
			p2.partial_load<2>(x + 2 * ldim);

			p0 = unpack_low(p0, p1);
			p2 = p2.swizzle<0,2,1,3>();

			col0.m_pk = merge_low (p0, p2);
			col1.m_pk = merge_high(p0, p2);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + 3, unaligned_t());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, int ldim, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + ldim, AlignT());
		}

		LSIMD_ENSURE_INLINE f32 trace() const
		{
			return col0.m_pk.e[0] + col1.m_pk.e[1];
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return col0.test_equal(r) && col1.test_equal(r + 3);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [3 x 2]:\n");

			std::printf("    col0 = ");
			col0.m_pk.dump(fmt);
			std::printf("\n");

			std::printf("    col1 = ");
			col1.m_pk.dump(fmt);
			std::printf("\n");
		}

	};


	template<>
	struct smat_core<f32, 3, 3>
	{
		typedef sse_vec<f32, 3> vec_t;

		vec_t col0;
		vec_t col1;
		vec_t col2;

		LSIMD_ENSURE_INLINE
		smat_core() { }

		LSIMD_ENSURE_INLINE
		smat_core( zero_t )
		: col0( zero_t() ), col1( zero_t() ), col2( zero_t() ){ }

		LSIMD_ENSURE_INLINE
		smat_core(const vec_t& v0, const vec_t& v1, const vec_t& v2 )
		: col0(v0), col1(v1), col2( v2 ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + 3, unaligned_t());
			col2.load(x + 6, unaligned_t());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, int ldim, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + ldim, AlignT());
			col2.load(x + 2 * ldim, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, AlignT)
		{
			_load_trans(x, x + 3, x + 6);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, int ldim, AlignT)
		{
			_load_trans(x, x + ldim, x + 2 * ldim);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + 3, unaligned_t());
			col2.store(x + 6, unaligned_t());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, int ldim, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + ldim, AlignT());
			col2.store(x + 2 * ldim, AlignT());
		}

		LSIMD_ENSURE_INLINE f32 trace() const
		{
			return col0.m_pk.e[0] + col1.m_pk.e[1] + col2.m_pk.e[2];
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return  col0.test_equal(r) &&
					col1.test_equal(r + 3) &&
					col2.test_equal(r + 6);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [3 x 3]:\n");

			std::printf("    col0 = ");
			col0.m_pk.dump(fmt);
			std::printf("\n");

			std::printf("    col1 = ");
			col1.m_pk.dump(fmt);
			std::printf("\n");

			std::printf("    col2 = ");
			col2.m_pk.dump(fmt);
			std::printf("\n");
		}

	private:
		LSIMD_ENSURE_INLINE
		void _load_trans(const f32 *r0, const f32 *r1, const f32 *r2)
		{
			sse_f32pk p0, p1, p2;

			p0.partial_load<2>(r0);
			p1.partial_load<2>(r1);
			p2.partial_load<2>(r2);

			sse_f32pk u0 = unpack_low(p0, p1);
			sse_f32pk u1 = p2.swizzle<0,2,1,3>();

			col0.m_pk = merge_low (u0, u1);
			col1.m_pk = merge_high(u0, u1);

			col2.m_pk.set(r0[2], r1[2], r2[2], 0.f);
		}
	};


	template<>
	struct smat_core<f32, 3, 4>
	{
		typedef sse_vec<f32, 3> vec_t;

		vec_t col0;
		vec_t col1;
		vec_t col2;
		vec_t col3;

		LSIMD_ENSURE_INLINE
		smat_core() { }

		LSIMD_ENSURE_INLINE
		smat_core( zero_t )
		: col0( zero_t() ), col1( zero_t() ), col2( zero_t() ), col3( zero_t() ){ }

		LSIMD_ENSURE_INLINE
		smat_core(const vec_t& v0, const vec_t& v1, const vec_t& v2, const vec_t& v3 )
		: col0(v0), col1(v1), col2( v2 ), col3( v3 ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + 3, unaligned_t());
			col2.load(x + 6, unaligned_t());
			col3.load(x + 9, unaligned_t());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, int ldim, AlignT)
		{
			const f32 *x2 = x + 2 * ldim;

			col0.load(x, AlignT());
			col1.load(x + ldim, AlignT());
			col2.load(x2, AlignT());
			col3.load(x2 + ldim, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, AlignT)
		{
			_load_trans(x, x + 4, x + 8, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, int ldim, AlignT)
		{
			_load_trans(x, x + ldim, x + 2 * ldim, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + 3, unaligned_t());
			col2.store(x + 6, unaligned_t());
			col3.store(x + 9, unaligned_t());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, int ldim, AlignT) const
		{
			f32 *x2 = x + 2 * ldim;

			col0.store(x, AlignT());
			col1.store(x + ldim, AlignT());
			col2.store(x2, AlignT());
			col3.store(x2 + ldim, AlignT());
		}

		LSIMD_ENSURE_INLINE f32 trace() const
		{
			return col0.m_pk.e[0] + col1.m_pk.e[1] + col2.m_pk.e[2];
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return  col0.test_equal(r) &&
					col1.test_equal(r + 3) &&
					col2.test_equal(r + 6) &&
					col3.test_equal(r + 9);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [3 x 4]:\n");

			std::printf("    col0 = ");
			col0.m_pk.dump(fmt);
			std::printf("\n");

			std::printf("    col1 = ");
			col1.m_pk.dump(fmt);
			std::printf("\n");

			std::printf("    col2 = ");
			col2.m_pk.dump(fmt);
			std::printf("\n");

			std::printf("    col3 = ");
			col3.m_pk.dump(fmt);
			std::printf("\n");
		}

	private:
		template<typename AlignT>
		LSIMD_ENSURE_INLINE
		void _load_trans(const f32 *r0, const f32 *r1, const f32 *r2, AlignT)
		{
			sse_f32pk p0(r0, AlignT());
			sse_f32pk p1(r1, AlignT());
			sse_f32pk p2(r2, AlignT());
			sse_f32pk pz = zero_t();

			sse_f32pk u0l = unpack_low(p0, p1);
			sse_f32pk u0h = unpack_high(p0, p1);
			sse_f32pk u1l = unpack_low(p2, pz);
			sse_f32pk u1h = unpack_high(p2, pz);

			col0.m_pk = merge_low (u0l, u1l);
			col1.m_pk = merge_high(u0l, u1l);
			col2.m_pk = merge_low (u0h, u1h);
			col3.m_pk = merge_high(u0h, u1h);
		}

	};



	/********************************************
	 *
	 *  specialized core classes for f32 4 x N
	 *
	 ********************************************/


	template<>
	struct smat_core<f32, 4, 2>
	{
		typedef sse_vec<f32, 4> vec_t;

		vec_t col0;
		vec_t col1;

		LSIMD_ENSURE_INLINE
		smat_core() { }

		LSIMD_ENSURE_INLINE
		smat_core( zero_t )
		: col0( zero_t() ), col1( zero_t() ) { }

		LSIMD_ENSURE_INLINE
		smat_core(const vec_t& v0, const vec_t& v1 )
		: col0(v0), col1(v1) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + 4, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, int ldim, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + ldim, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, AlignT)
		{
			sse_f32pk p01, p23;

			p01.load(x, AlignT());
			p23.load(x + 4, AlignT());

			sse_f32pk u0 = unpack_low(p01, p23);
			sse_f32pk u1 = unpack_high(p01, p23);

			col0.m_pk = unpack_low(u0, u1);
			col1.m_pk = unpack_high(u0, u1);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, int ldim, AlignT)
		{
			const f32 *x2 = x + 2 * ldim;

			sse_f32pk p0, p1, p2, p3;

			p0.partial_load<2>(x);
			p1.partial_load<2>(x + ldim);
			p2.partial_load<2>(x2);
			p3.partial_load<2>(x2 + ldim);

			sse_f32pk u0 = unpack_low(p0, p2);
			sse_f32pk u1 = unpack_low(p1, p3);

			col0.m_pk = unpack_low(u0, u1);
			col1.m_pk = unpack_high(u0, u1);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + 4, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, int ldim, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + ldim, AlignT());
		}

		LSIMD_ENSURE_INLINE f32 trace() const
		{
			return col0.m_pk.e[0] + col1.m_pk.e[1];
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return  col0.test_equal(r) &&
					col1.test_equal(r + 4);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [4 x 2]:\n");

			std::printf("    col0 = ");
			col0.m_pk.dump(fmt);
			std::printf("\n");

			std::printf("    col1 = ");
			col1.m_pk.dump(fmt);
			std::printf("\n");
		}
	};


	template<>
	struct smat_core<f32, 4, 3>
	{
		typedef sse_vec<f32, 4> vec_t;

		vec_t col0;
		vec_t col1;
		vec_t col2;

		LSIMD_ENSURE_INLINE
		smat_core() { }

		LSIMD_ENSURE_INLINE
		smat_core( zero_t )
		: col0( zero_t() ), col1( zero_t() ), col2( zero_t() ){ }

		LSIMD_ENSURE_INLINE
		smat_core(const vec_t& v0, const vec_t& v1, const vec_t& v2 )
		: col0(v0), col1(v1), col2( v2 ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + 4, AlignT());
			col2.load(x + 8, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, int ldim, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + ldim, AlignT());
			col2.load(x + 2 * ldim, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, AlignT)
		{
			_load_trans(x, x + 3, x + 6, x + 9);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, int ldim, AlignT)
		{
			const f32 *x2 = x + 2 * ldim;
			_load_trans(x, x + ldim, x2, x2 + ldim);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + 4, AlignT());
			col2.store(x + 8, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, int ldim, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + ldim, AlignT());
			col2.store(x + 2 * ldim, AlignT());
		}

		LSIMD_ENSURE_INLINE f32 trace() const
		{
			return col0.m_pk.e[0] + col1.m_pk.e[1] + col2.m_pk.e[2];
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return  col0.test_equal(r) &&
					col1.test_equal(r + 4) &&
					col2.test_equal(r + 8);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [4 x 3]:\n");

			std::printf("    col0 = ");
			col0.m_pk.dump(fmt);
			std::printf("\n");

			std::printf("    col1 = ");
			col1.m_pk.dump(fmt);
			std::printf("\n");

			std::printf("    col2 = ");
			col2.m_pk.dump(fmt);
			std::printf("\n");
		}

	private:
		LSIMD_ENSURE_INLINE
		void _load_trans(const f32 *r0, const f32 *r1, const f32 *r2, const f32 * r3)
		{
			sse_f32pk p0, p1, p2, p3;

			p0.partial_load<2>(r0);
			p1.partial_load<2>(r1);
			p2.partial_load<2>(r2);
			p3.partial_load<3>(r3);

			p0 = unpack_low(p0, p1);
			p2 = unpack_low(p2, p3);

			col0.m_pk = merge_low (p0, p2);
			col1.m_pk = merge_high(p0, p2);

			col2.m_pk.set(r0[2], r1[2], r2[2], r3[2]);
		}

	};


	template<>
	struct smat_core<f32, 4, 4>
	{
		typedef sse_vec<f32, 4> vec_t;

		vec_t col0;
		vec_t col1;
		vec_t col2;
		vec_t col3;

		LSIMD_ENSURE_INLINE
		smat_core() { }

		LSIMD_ENSURE_INLINE
		smat_core( zero_t )
		: col0( zero_t() ), col1( zero_t() ), col2( zero_t() ), col3( zero_t() ){ }

		LSIMD_ENSURE_INLINE
		smat_core(const vec_t& v0, const vec_t& v1, const vec_t& v2, const vec_t& v3 )
		: col0(v0), col1(v1), col2( v2 ), col3( v3 ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + 4, AlignT());
			col2.load(x + 8, AlignT());
			col3.load(x + 12, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, int ldim, AlignT)
		{
			const f32 *x2 = x + 2 * ldim;

			col0.load(x, AlignT());
			col1.load(x + ldim, AlignT());
			col2.load(x2, AlignT());
			col3.load(x2 + ldim, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, AlignT)
		{
			_load_trans(x, x + 4, x + 8, x + 12, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, int ldim, AlignT)
		{
			const f32 *x2 = x + 2 * ldim;
			_load_trans(x, x + ldim, x2, x2 + ldim, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + 4, AlignT());
			col2.store(x + 8, AlignT());
			col3.store(x + 12, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, int ldim, AlignT) const
		{
			f32 *x2 = x + 2 * ldim;

			col0.store(x, AlignT());
			col1.store(x + ldim, AlignT());
			col2.store(x2, AlignT());
			col3.store(x2 + ldim, AlignT());
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return  col0.test_equal(r) &&
					col1.test_equal(r + 4) &&
					col2.test_equal(r + 8) &&
					col3.test_equal(r + 12);
		}

		LSIMD_ENSURE_INLINE f32 trace() const
		{
			return col0.m_pk.e[0] + col1.m_pk.e[1] + col2.m_pk.e[2] + col3.m_pk.e[3];
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [4 x 4]:\n");

			std::printf("    col0 = ");
			col0.m_pk.dump(fmt);
			std::printf("\n");

			std::printf("    col1 = ");
			col1.m_pk.dump(fmt);
			std::printf("\n");

			std::printf("    col2 = ");
			col2.m_pk.dump(fmt);
			std::printf("\n");

			std::printf("    col3 = ");
			col3.m_pk.dump(fmt);
			std::printf("\n");
		}

	private:
		template<typename AlignT>
		LSIMD_ENSURE_INLINE
		void _load_trans(const f32 *r0, const f32 *r1, const f32 *r2, const f32 *r3, AlignT)
		{
			sse_f32pk p0, p1, p2, p3;

			p0.load(r0, AlignT());
			p1.load(r1, AlignT());
			p2.load(r2, AlignT());
			p3.load(r3, AlignT());

			sse_f32pk u0l = unpack_low (p0, p1);
			sse_f32pk u0h = unpack_high(p0, p1);
			sse_f32pk u1l = unpack_low (p2, p3);
			sse_f32pk u1h = unpack_high(p2, p3);

			col0.m_pk = merge_low (u0l, u1l);
			col1.m_pk = merge_high(u0l, u1l);
			col2.m_pk = merge_low (u0h, u1h);
			col3.m_pk = merge_high(u0h, u1h);
		}

	};


	/********************************************
	 *
	 *  specialized core classes for f64 2 x N
	 *
	 ********************************************/

	template<>
	struct smat_core<f64, 2, 2>
	{
		typedef sse_vec<f64, 2> vec_t;

		vec_t col0;
		vec_t col1;

		LSIMD_ENSURE_INLINE
		smat_core() { }

		LSIMD_ENSURE_INLINE
		smat_core( zero_t )
		: col0( zero_t() ), col1( zero_t() ) { }

		LSIMD_ENSURE_INLINE
		smat_core(const vec_t& v0, const vec_t& v1 )
		: col0(v0), col1(v1) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + 2, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, int ldim, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + ldim, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, AlignT)
		{
			sse_f64pk pr0(x, AlignT());
			sse_f64pk pr1(x + 2, AlignT());

			col0.m_pk = unpack_low(pr0, pr1);
			col1.m_pk = unpack_high(pr0, pr1);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, int ldim, AlignT)
		{
			sse_f64pk pr0(x, AlignT());
			sse_f64pk pr1(x + ldim, AlignT());

			col0.m_pk = unpack_low(pr0, pr1);
			col1.m_pk = unpack_high(pr0, pr1);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + 2, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, int ldim, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + ldim, AlignT());
		}

		LSIMD_ENSURE_INLINE f64 trace() const
		{
			return col0.m_pk.e[0] + col1.m_pk.e[1];
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return  col0.test_equal(r) &&
					col1.test_equal(r + 2);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [2 x 2]:\n");

			std::printf("    col0 = ");
			col0.m_pk.dump(fmt);
			std::printf("\n");

			std::printf("    col1 = ");
			col1.m_pk.dump(fmt);
			std::printf("\n");
		}
	};


	template<>
	struct smat_core<f64, 2, 3>
	{
		typedef sse_vec<f64, 2> vec_t;

		vec_t col0;
		vec_t col1;
		vec_t col2;

		LSIMD_ENSURE_INLINE
		smat_core() { }

		LSIMD_ENSURE_INLINE
		smat_core( zero_t )
		: col0( zero_t() ), col1( zero_t() ), col2( zero_t() ){ }

		LSIMD_ENSURE_INLINE
		smat_core(const vec_t& v0, const vec_t& v1, const vec_t& v2 )
		: col0(v0), col1(v1), col2( v2 ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + 2, AlignT());
			col2.load(x + 4, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, int ldim, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + ldim, AlignT());
			col2.load(x + ldim * 2, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, AlignT)
		{
			sse_f64pk pa(x, AlignT());
			sse_f64pk pb(x + 2, AlignT());
			sse_f64pk pc(x + 4, AlignT());

			col0.m_pk = shuffle<0, 1>(pa, pb);
			col1.m_pk = shuffle<1, 0>(pa, pc);
			col2.m_pk = shuffle<0, 1>(pb, pc);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, int ldim, AlignT)
		{
			sse_f64pk p0l, p0h, p1l, p1h;

			p0l.load(x, AlignT());
			p0h.partial_load<1>(x + 2);

			p1l.load(x + ldim, AlignT());
			p1h.partial_load<1>(x + ldim + 2);

			col0.m_pk = unpack_low(p0l, p1l);
			col1.m_pk = unpack_high(p0l, p1l);
			col2.m_pk = unpack_low(p0h, p1h);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + 2, AlignT());
			col2.store(x + 4, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, int ldim, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + ldim, AlignT());
			col2.store(x + ldim * 2, AlignT());
		}

		LSIMD_ENSURE_INLINE f64 trace() const
		{
			return col0.m_pk.e[0] + col1.m_pk.e[1];
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return  col0.test_equal(r) &&
					col1.test_equal(r + 2) &&
					col2.test_equal(r + 4);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [2 x 3]:\n");

			std::printf("    col0 = ");
			col0.m_pk.dump(fmt);
			std::printf("\n");

			std::printf("    col1 = ");
			col1.m_pk.dump(fmt);
			std::printf("\n");

			std::printf("    col2 = ");
			col2.m_pk.dump(fmt);
			std::printf("\n");
		}
	};


	template<>
	struct smat_core<f64, 2, 4>
	{
		typedef sse_vec<f64, 2> vec_t;

		vec_t col0;
		vec_t col1;
		vec_t col2;
		vec_t col3;

		LSIMD_ENSURE_INLINE
		smat_core() { }

		LSIMD_ENSURE_INLINE
		smat_core( zero_t )
		: col0( zero_t() ), col1( zero_t() ), col2( zero_t() ), col3( zero_t() ){ }

		LSIMD_ENSURE_INLINE
		smat_core(const vec_t& v0, const vec_t& v1, const vec_t& v2, const vec_t& v3 )
		: col0(v0), col1(v1), col2( v2 ), col3( v3 ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + 2, AlignT());
			col2.load(x + 4, AlignT());
			col3.load(x + 6, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, int ldim, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + ldim, AlignT());
			col2.load(x + ldim * 2, AlignT());
			col3.load(x + ldim * 3, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, AlignT)
		{
			sse_f64pk p0l(x, AlignT());
			sse_f64pk p0h(x + 2, AlignT());
			sse_f64pk p1l(x + 4, AlignT());
			sse_f64pk p1h(x + 6, AlignT());

			col0.m_pk = unpack_low (p0l, p1l);
			col1.m_pk = unpack_high(p0l, p1l);
			col2.m_pk = unpack_low (p0h, p1h);
			col3.m_pk = unpack_high(p0h, p1h);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, int ldim, AlignT)
		{
			const f64 *x1 = x + ldim;

			sse_f64pk p0l(x, AlignT());
			sse_f64pk p0h(x + 2, AlignT());
			sse_f64pk p1l(x1, AlignT());
			sse_f64pk p1h(x1 + 2, AlignT());

			col0.m_pk = unpack_low (p0l, p1l);
			col1.m_pk = unpack_high(p0l, p1l);
			col2.m_pk = unpack_low (p0h, p1h);
			col3.m_pk = unpack_high(p0h, p1h);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + 2, AlignT());
			col2.store(x + 4, AlignT());
			col3.store(x + 6, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, int ldim, AlignT) const
		{
			f64 *x2 = x + 2 * ldim;

			col0.store(x, AlignT());
			col1.store(x + ldim, AlignT());
			col2.store(x2, AlignT());
			col3.store(x2 + ldim, AlignT());
		}

		LSIMD_ENSURE_INLINE f64 trace() const
		{
			return col0.m_pk.e[0] + col1.m_pk.e[1];
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return  col0.test_equal(r) &&
					col1.test_equal(r + 2) &&
					col2.test_equal(r + 4) &&
					col3.test_equal(r + 6);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [2 x 4]:\n");

			std::printf("    col0 = ");
			col0.m_pk.dump(fmt);
			std::printf("\n");

			std::printf("    col1 = ");
			col1.m_pk.dump(fmt);
			std::printf("\n");

			std::printf("    col2 = ");
			col2.m_pk.dump(fmt);
			std::printf("\n");

			std::printf("    col3 = ");
			col3.m_pk.dump(fmt);
			std::printf("\n");
		}

	};



	/********************************************
	 *
	 *  specialized core classes for f64 3 x N
	 *
	 ********************************************/

	template<>
	struct smat_core<f64, 3, 2>
	{
		typedef sse_vec<f64, 3> vec_t;

		vec_t col0;
		vec_t col1;

		LSIMD_ENSURE_INLINE
		smat_core() { }

		LSIMD_ENSURE_INLINE
		smat_core( zero_t )
		: col0( zero_t() ), col1( zero_t() ) { }

		LSIMD_ENSURE_INLINE
		smat_core(const vec_t& v0, const vec_t& v1 )
		: col0(v0), col1(v1) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + 3, unaligned_t());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, int ldim, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + ldim, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, AlignT)
		{
			sse_f64pk pr0(x, AlignT());
			sse_f64pk pr1(x + 2, AlignT());
			sse_f64pk pr2(x + 4, AlignT());

			_load_trans(pr0, pr1, pr2);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, int ldim, AlignT)
		{
			sse_f64pk pr0(x, AlignT());
			sse_f64pk pr1(x + ldim, AlignT());
			sse_f64pk pr2(x + ldim * 2, AlignT());

			_load_trans(pr0, pr1, pr2);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + 3, unaligned_t());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, int ldim, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + ldim, AlignT());
		}

		LSIMD_ENSURE_INLINE f64 trace() const
		{
			return col0.m_pk0.e[0] + col1.m_pk0.e[1];
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return  col0.test_equal(r) &&
					col1.test_equal(r + 3);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [3 x 2]:\n");

			std::printf("    col0 = ");
			col0.m_pk0.dump(fmt);
			std::printf(" ");
			col0.m_pk1.dump(fmt);
			std::printf("\n");

			std::printf("    col1 = ");
			col1.m_pk0.dump(fmt);
			std::printf(" ");
			col1.m_pk1.dump(fmt);
			std::printf("\n");
		}

	private:
		LSIMD_ENSURE_INLINE
		void _load_trans(const sse_f64pk& pr0, const sse_f64pk& pr1, const sse_f64pk& pr2)
		{
			sse_f64pk z = zero_t();

			col0.m_pk0 = unpack_low (pr0, pr1);
			col0.m_pk1 = unpack_low (pr2, z);
			col1.m_pk0 = unpack_high(pr0, pr1);
			col1.m_pk1 = unpack_high(pr2, z);
		}
	};


	template<>
	struct smat_core<f64, 3, 3>
	{
		typedef sse_vec<f64, 3> vec_t;

		vec_t col0;
		vec_t col1;
		vec_t col2;

		LSIMD_ENSURE_INLINE
		smat_core() { }

		LSIMD_ENSURE_INLINE
		smat_core( zero_t )
		: col0( zero_t() ), col1( zero_t() ), col2( zero_t() ){ }

		LSIMD_ENSURE_INLINE
		smat_core(const vec_t& v0, const vec_t& v1, const vec_t& v2 )
		: col0(v0), col1(v1), col2( v2 ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + 3, unaligned_t());
			col2.load(x + 6, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, int ldim, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + ldim, AlignT());
			col2.load(x + ldim * 2, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, AlignT)
		{
			sse_f64pk l0, l1, l2;
			sse_f64pk h0, h1, h2;

			l0.load(x, AlignT());
			h0.partial_load<1>(x + 2);

			l1.load(x + 3, unaligned_t());
			h1.partial_load<1>(x + 5);

			l2.load(x + 6, AlignT());
			h2.partial_load<1>(x + 8);

			sse_f64pk z = zero_t();

			col0.m_pk0 = unpack_low(l0, l1);
			col0.m_pk1 = unpack_low(l2, z);

			col1.m_pk0 = unpack_high(l0, l1);
			col1.m_pk1 = unpack_high(l2, z);

			col2.m_pk0 = unpack_low(h0, h1);
			col2.m_pk1 = unpack_low(h2, z);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, int ldim, AlignT)
		{
			sse_f64pk l0, l1, l2;
			sse_f64pk h0, h1, h2;

			const f64 *x1 = x + ldim;
			const f64 *x2 = x1 + ldim;

			l0.load(x, AlignT());
			h0.partial_load<1>(x + 2);

			l1.load(x1, AlignT());
			h1.partial_load<1>(x1 + 2);

			l2.load(x2, AlignT());
			h2.partial_load<1>(x2 + 2);

			sse_f64pk z = zero_t();

			col0.m_pk0 = unpack_low(l0, l1);
			col0.m_pk1 = unpack_low(l2, z);

			col1.m_pk0 = unpack_high(l0, l1);
			col1.m_pk1 = unpack_high(l2, z);

			col2.m_pk0 = unpack_low(h0, h1);
			col2.m_pk1 = unpack_low(h2, z);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + 3, unaligned_t());
			col2.store(x + 6, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, int ldim, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + ldim, AlignT());
			col2.store(x + ldim * 2, AlignT());
		}

		LSIMD_ENSURE_INLINE f64 trace() const
		{
			return col0.m_pk0.e[0] + col1.m_pk0.e[1] + col2.m_pk1.e[0];
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return  col0.test_equal(r) &&
					col1.test_equal(r + 3) &&
					col2.test_equal(r + 6);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [3 x 3]:\n");

			std::printf("    col0 = ");
			col0.m_pk0.dump(fmt);
			std::printf(" ");
			col0.m_pk1.dump(fmt);
			std::printf("\n");

			std::printf("    col1 = ");
			col1.m_pk0.dump(fmt);
			std::printf(" ");
			col1.m_pk1.dump(fmt);
			std::printf("\n");

			std::printf("    col2 = ");
			col2.m_pk0.dump(fmt);
			std::printf(" ");
			col2.m_pk1.dump(fmt);
			std::printf("\n");
		}
	};


	template<>
	struct smat_core<f64, 3, 4>
	{
		typedef sse_vec<f64, 3> vec_t;

		vec_t col0;
		vec_t col1;
		vec_t col2;
		vec_t col3;

		LSIMD_ENSURE_INLINE
		smat_core() { }

		LSIMD_ENSURE_INLINE
		smat_core( zero_t )
		: col0( zero_t() ), col1( zero_t() ), col2( zero_t() ), col3( zero_t() ){ }

		LSIMD_ENSURE_INLINE
		smat_core(const vec_t& v0, const vec_t& v1, const vec_t& v2, const vec_t& v3 )
		: col0(v0), col1(v1), col2( v2 ), col3( v3 ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + 3, unaligned_t());
			col2.load(x + 6, AlignT());
			col3.load(x + 9, unaligned_t());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, int ldim, AlignT)
		{
			const f64 *x2 = x + ldim * 2;

			col0.load(x, AlignT());
			col1.load(x + ldim, AlignT());
			col2.load(x2, AlignT());
			col3.load(x2 + ldim, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, AlignT)
		{
			_load_trans(x, x + 4, x + 8, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, int ldim, AlignT)
		{
			_load_trans(x, x + ldim, x + ldim * 2, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + 3, unaligned_t());
			col2.store(x + 6, AlignT());
			col3.store(x + 9, unaligned_t());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, int ldim, AlignT) const
		{
			f64 *x2 = x + ldim * 2;

			col0.store(x, AlignT());
			col1.store(x + ldim, AlignT());
			col2.store(x2, AlignT());
			col3.store(x2 + ldim, AlignT());
		}

		LSIMD_ENSURE_INLINE f64 trace() const
		{
			return col0.m_pk0.e[0] + col1.m_pk0.e[1] + col2.m_pk1.e[0];
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return  col0.test_equal(r) &&
					col1.test_equal(r + 3) &&
					col2.test_equal(r + 6) &&
					col3.test_equal(r + 9);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [3 x 4]:\n");

			std::printf("    col0 = ");
			col0.m_pk0.dump(fmt);
			std::printf(" ");
			col0.m_pk1.dump(fmt);
			std::printf("\n");

			std::printf("    col1 = ");
			col1.m_pk0.dump(fmt);
			std::printf(" ");
			col1.m_pk1.dump(fmt);
			std::printf("\n");

			std::printf("    col2 = ");
			col2.m_pk0.dump(fmt);
			std::printf(" ");
			col2.m_pk1.dump(fmt);
			std::printf("\n");

			std::printf("    col3 = ");
			col3.m_pk0.dump(fmt);
			std::printf(" ");
			col3.m_pk1.dump(fmt);
			std::printf("\n");
		}

	private:
		template<typename AlignT>
		LSIMD_ENSURE_INLINE
		void _load_trans(const f64 *r0, const f64 *r1, const f64 *r2, AlignT)
		{
			sse_f64pk l0(r0,     AlignT());
			sse_f64pk h0(r0 + 2, AlignT());
			sse_f64pk l1(r1,     AlignT());
			sse_f64pk h1(r1 + 2, AlignT());
			sse_f64pk l2(r2,     AlignT());
			sse_f64pk h2(r2 + 2, AlignT());

			sse_f64pk z = zero_t();

			col0.m_pk0 = unpack_low(l0, l1);
			col0.m_pk1 = unpack_low(l2, z);

			col1.m_pk0 = unpack_high(l0, l1);
			col1.m_pk1 = unpack_high(l2, z);

			col2.m_pk0 = unpack_low(h0, h1);
			col2.m_pk1 = unpack_low(h2, z);

			col3.m_pk0 = unpack_high(h0, h1);
			col3.m_pk1 = unpack_high(h2, z);
		}

	};



	/********************************************
	 *
	 *  specialized core classes for f64 4 x N
	 *
	 ********************************************/

	template<>
	struct smat_core<f64, 4, 2>
	{
		typedef sse_vec<f64, 4> vec_t;

		vec_t col0;
		vec_t col1;

		LSIMD_ENSURE_INLINE
		smat_core() { }

		LSIMD_ENSURE_INLINE
		smat_core( zero_t )
		: col0( zero_t() ), col1( zero_t() ) { }

		LSIMD_ENSURE_INLINE
		smat_core(const vec_t& v0, const vec_t& v1 )
		: col0(v0), col1(v1) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + 4, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, int ldim, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + ldim, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, AlignT)
		{
			sse_f64pk pr0(x, AlignT());
			sse_f64pk pr1(x + 2, AlignT());
			sse_f64pk pr2(x + 4, AlignT());
			sse_f64pk pr3(x + 6, AlignT());

			_load_trans(pr0, pr1, pr2, pr3);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, int ldim, AlignT)
		{
			const f64 *x2 = x + ldim * 2;

			sse_f64pk pr0(x, AlignT());
			sse_f64pk pr1(x + ldim, AlignT());
			sse_f64pk pr2(x2, AlignT());
			sse_f64pk pr3(x2 + ldim, AlignT());

			_load_trans(pr0, pr1, pr2, pr3);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + 4, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, int ldim, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + ldim, AlignT());
		}

		LSIMD_ENSURE_INLINE f64 trace() const
		{
			return col0.m_pk0.e[0] + col1.m_pk0.e[1];
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return  col0.test_equal(r) &&
					col1.test_equal(r + 4);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [4 x 2]:\n");

			std::printf("    col0 = ");
			col0.m_pk0.dump(fmt);
			std::printf(" ");
			col0.m_pk1.dump(fmt);
			std::printf("\n");

			std::printf("    col1 = ");
			col1.m_pk0.dump(fmt);
			std::printf(" ");
			col1.m_pk1.dump(fmt);
			std::printf("\n");
		}

	private:
		LSIMD_ENSURE_INLINE
		void _load_trans(const sse_f64pk& pr0, const sse_f64pk& pr1,
			const sse_f64pk& pr2, const sse_f64pk& pr3)
		{
			col0.m_pk0 = unpack_low (pr0, pr1);
			col0.m_pk1 = unpack_low (pr2, pr3);
			col1.m_pk0 = unpack_high(pr0, pr1);
			col1.m_pk1 = unpack_high(pr2, pr3);
		}
	};


	template<>
	struct smat_core<f64, 4, 3>
	{
		typedef sse_vec<f64, 4> vec_t;

		vec_t col0;
		vec_t col1;
		vec_t col2;

		LSIMD_ENSURE_INLINE
		smat_core() { }

		LSIMD_ENSURE_INLINE
		smat_core( zero_t )
		: col0( zero_t() ), col1( zero_t() ), col2( zero_t() ){ }

		LSIMD_ENSURE_INLINE
		smat_core(const vec_t& v0, const vec_t& v1, const vec_t& v2 )
		: col0(v0), col1(v1), col2( v2 ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + 4, AlignT());
			col2.load(x + 8, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, int ldim, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + ldim, AlignT());
			col2.load(x + ldim * 2, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, AlignT)
		{
			sse_f64pk l0, l1, l2, l3;
			sse_f64pk h0, h1, h2, h3;

			l0.load(x, AlignT());
			h0.partial_load<1>(x + 2);

			l1.load(x + 3, unaligned_t());
			h1.partial_load<1>(x + 5);

			l2.load(x + 6, AlignT());
			h2.partial_load<1>(x + 8);

			l3.load(x + 9, unaligned_t());
			h3.partial_load<1>(x + 11);

			col0.m_pk0 = unpack_low(l0, l1);
			col0.m_pk1 = unpack_low(l2, l3);

			col1.m_pk0 = unpack_high(l0, l1);
			col1.m_pk1 = unpack_high(l2, l3);

			col2.m_pk0 = unpack_low(h0, h1);
			col2.m_pk1 = unpack_low(h2, h3);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, int ldim, AlignT)
		{
			sse_f64pk l0, l1, l2, l3;
			sse_f64pk h0, h1, h2, h3;

			const f64 *x1 = x + ldim;
			const f64 *x2 = x1 + ldim;
			const f64 *x3 = x2 + ldim;

			l0.load(x, AlignT());
			h0.partial_load<1>(x + 2);

			l1.load(x1, AlignT());
			h1.partial_load<1>(x1 + 2);

			l2.load(x2, AlignT());
			h2.partial_load<1>(x2 + 2);

			l3.load(x3, AlignT());
			h3.partial_load<1>(x3 + 2);

			col0.m_pk0 = unpack_low(l0, l1);
			col0.m_pk1 = unpack_low(l2, l3);

			col1.m_pk0 = unpack_high(l0, l1);
			col1.m_pk1 = unpack_high(l2, l3);

			col2.m_pk0 = unpack_low(h0, h1);
			col2.m_pk1 = unpack_low(h2, h3);
		}


		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + 4, AlignT());
			col2.store(x + 8, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, int ldim, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + ldim, AlignT());
			col2.store(x + ldim * 2, AlignT());
		}

		LSIMD_ENSURE_INLINE f64 trace() const
		{
			return col0.m_pk0.e[0] + col1.m_pk0.e[1] + col2.m_pk1.e[0];
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return  col0.test_equal(r) &&
					col1.test_equal(r + 4) &&
					col2.test_equal(r + 8);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [4 x 3]:\n");

			std::printf("    col0 = ");
			col0.m_pk0.dump(fmt);
			std::printf(" ");
			col0.m_pk1.dump(fmt);
			std::printf("\n");

			std::printf("    col1 = ");
			col1.m_pk0.dump(fmt);
			std::printf(" ");
			col1.m_pk1.dump(fmt);
			std::printf("\n");

			std::printf("    col2 = ");
			col2.m_pk0.dump(fmt);
			std::printf(" ");
			col2.m_pk1.dump(fmt);
			std::printf("\n");
		}
	};


	template<>
	struct smat_core<f64, 4, 4>
	{
		typedef sse_vec<f64, 4> vec_t;

		vec_t col0;
		vec_t col1;
		vec_t col2;
		vec_t col3;

		LSIMD_ENSURE_INLINE
		smat_core() { }

		LSIMD_ENSURE_INLINE
		smat_core( zero_t )
		: col0( zero_t() ), col1( zero_t() ), col2( zero_t() ), col3( zero_t() ){ }

		LSIMD_ENSURE_INLINE
		smat_core(const vec_t& v0, const vec_t& v1, const vec_t& v2, const vec_t& v3 )
		: col0(v0), col1(v1), col2( v2 ), col3( v3 ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, AlignT)
		{
			col0.load(x, AlignT());
			col1.load(x + 4, AlignT());
			col2.load(x + 8, AlignT());
			col3.load(x + 12, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, int ldim, AlignT)
		{
			const f64 *x2 = x + ldim * 2;

			col0.load(x, AlignT());
			col1.load(x + ldim, AlignT());
			col2.load(x2, AlignT());
			col3.load(x2 + ldim, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, AlignT)
		{
			_load_trans(x, x + 4, x + 8, x + 12, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, int ldim, AlignT)
		{
			const f64 *x2 = x + ldim * 2;
			_load_trans(x, x + ldim, x2, x2 + ldim, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, AlignT) const
		{
			col0.store(x, AlignT());
			col1.store(x + 4, AlignT());
			col2.store(x + 8, AlignT());
			col3.store(x + 12, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, int ldim, AlignT) const
		{
			f64 *x2 = x + ldim * 2;

			col0.store(x, AlignT());
			col1.store(x + ldim, AlignT());
			col2.store(x2, AlignT());
			col3.store(x2 + ldim, AlignT());
		}

		LSIMD_ENSURE_INLINE f64 trace() const
		{
			return col0.m_pk0.e[0] + col1.m_pk0.e[1] + col2.m_pk1.e[0] + col3.m_pk1.e[1];
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return  col0.test_equal(r) &&
					col1.test_equal(r + 4) &&
					col2.test_equal(r + 8) &&
					col3.test_equal(r + 12);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [4 x 4]:\n");

			std::printf("    col0 = ");
			col0.m_pk0.dump(fmt);
			std::printf(" ");
			col0.m_pk1.dump(fmt);
			std::printf("\n");

			std::printf("    col1 = ");
			col1.m_pk0.dump(fmt);
			std::printf(" ");
			col1.m_pk1.dump(fmt);
			std::printf("\n");

			std::printf("    col2 = ");
			col2.m_pk0.dump(fmt);
			std::printf(" ");
			col2.m_pk1.dump(fmt);
			std::printf("\n");

			std::printf("    col3 = ");
			col3.m_pk0.dump(fmt);
			std::printf(" ");
			col3.m_pk1.dump(fmt);
			std::printf("\n");
		}

	private:
		template<typename AlignT>
		LSIMD_ENSURE_INLINE
		void _load_trans(const f64 *r0, const f64 *r1, const f64 *r2, const f64 *r3, AlignT)
		{
			sse_f64pk l0(r0,     AlignT());
			sse_f64pk h0(r0 + 2, AlignT());
			sse_f64pk l1(r1,     AlignT());
			sse_f64pk h1(r1 + 2, AlignT());
			sse_f64pk l2(r2,     AlignT());
			sse_f64pk h2(r2 + 2, AlignT());
			sse_f64pk l3(r3,     AlignT());
			sse_f64pk h3(r3 + 2, AlignT());

			col0.m_pk0 = unpack_low(l0, l1);
			col0.m_pk1 = unpack_low(l2, l3);

			col1.m_pk0 = unpack_high(l0, l1);
			col1.m_pk1 = unpack_high(l2, l3);

			col2.m_pk0 = unpack_low(h0, h1);
			col2.m_pk1 = unpack_low(h2, h3);

			col3.m_pk0 = unpack_high(h0, h1);
			col3.m_pk1 = unpack_high(h2, h3);
		}

	};



} }

#endif /* SSE_MULTICOL_BASE_H_ */
