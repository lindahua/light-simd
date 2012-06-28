/**
 * @file sse_mat_matmul_bits.h
 *
 * The internal implementation for SSE-based small matrix multiplication
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_MAT_MATMUL_BITS_H_
#define LSIMD_SSE_MAT_MATMUL_BITS_H_

#include "sse_mat_comp_bits.h"

namespace lsimd { namespace sse {

	template<typename T, int M, int K, int N> struct mtimes_op;

	/********************************************
	 *
	 *  Generic implementation
	 *
	 ********************************************/

	template<typename T, int M, int K>
	struct mtimes_op<T, M, K, 2>
	{
		LSIMD_ENSURE_INLINE
		static void run(
				const smat_core<T, M, K>& A,
				const smat_core<T, K, 2>& B,
				      smat_core<T, M, 2>& C)
		{
			C.col0 = transform(A, B.col0);
			C.col1 = transform(A, B.col1);
		}
	};

	template<typename T, int M, int K>
	struct mtimes_op<T, M, K, 3>
	{
		LSIMD_ENSURE_INLINE
		static void run(
				const smat_core<T, M, K>& A,
				const smat_core<T, K, 3>& B,
				      smat_core<T, M, 3>& C)
		{
			C.col0 = transform(A, B.col0);
			C.col1 = transform(A, B.col1);
			C.col2 = transform(A, B.col2);
		}
	};

	template<typename T, int M, int K>
	struct mtimes_op<T, M, K, 4>
	{
		LSIMD_ENSURE_INLINE
		static void run(
				const smat_core<T, M, K>& A,
				const smat_core<T, K, 4>& B,
				      smat_core<T, M, 4>& C)
		{
			C.col0 = transform(A, B.col0);
			C.col1 = transform(A, B.col1);
			C.col2 = transform(A, B.col2);
			C.col3 = transform(A, B.col3);
		}
	};


	/******************************************************
	 *
	 *  Specialized implementation for (2x2) * (2xN)
	 *
	 ******************************************************/

	template<>
	struct mtimes_op<f32, 2, 2, 2>
	{
		LSIMD_ENSURE_INLINE
		static void run(
				const smat_core<f32, 2, 2>& A,
				const smat_core<f32, 2, 2>& B,
				      smat_core<f32, 2, 2>& C)
		{
			C.col01_pk = (A.col01_pk.dup_low() * B.col01_pk.dup2_low()) +
					(A.col01_pk.dup_high() * B.col01_pk.dup2_high());

		}
	};

	template<>
	struct mtimes_op<f32, 2, 2, 3>
	{
		LSIMD_ENSURE_INLINE
		static void run(
				const smat_core<f32, 2, 2>& A,
				const smat_core<f32, 2, 3>& B,
				      smat_core<f32, 2, 3>& C)
		{
			sse_f32pk ac0 = A.col01_pk.dup_low();
			sse_f32pk ac1 = A.col01_pk.dup_high();

			C.col01_pk = (ac0 * B.col01_pk.dup2_low()) + (ac1 * B.col01_pk.dup2_high());
			C.col2z_pk = (ac0 * B.col2z_pk.dup2_low()) + (ac1 * B.col2z_pk.dup2_high());
		}
	};

	template<>
	struct mtimes_op<f32, 2, 2, 4>
	{
		LSIMD_ENSURE_INLINE
		static void run(const smat_core<f32, 2, 2>& A,
				 const smat_core<f32, 2, 4>& B,
				       smat_core<f32, 2, 4>& C)
		{
			sse_f32pk ac0 = A.col01_pk.dup_low();
			sse_f32pk ac1 = A.col01_pk.dup_high();

			C.col01_pk = (ac0 * B.col01_pk.dup2_low()) + (ac1 * B.col01_pk.dup2_high());
			C.col23_pk = (ac0 * B.col23_pk.dup2_low()) + (ac1 * B.col23_pk.dup2_high());
		}
	};



	/******************************************************
	 *
	 *  Specialized implementation for (2x3) * (3xN)
	 *
	 ******************************************************/

	template<>
	struct mtimes_op<f32, 2, 3, 2>
	{
		LSIMD_ENSURE_INLINE
		static void run(
				const smat_core<f32, 2, 3>& A,
				const smat_core<f32, 3, 2>& B,
				      smat_core<f32, 2, 2>& C)
		{
			sse_f32pk ac0 = A.col01_pk.dup_low();
			sse_f32pk ac1 = A.col01_pk.dup_high();
			sse_f32pk ac2 = A.col2z_pk.dup_low();

			sse_f32pk bc0 = B.col0.m_pk;
			sse_f32pk bc1 = B.col1.m_pk;

			C.col01_pk =
					(ac0 * shuffle<0,0,0,0>(bc0, bc1)) +
					(ac1 * shuffle<1,1,1,1>(bc0, bc1)) +
					(ac2 * shuffle<2,2,2,2>(bc0, bc1));
		}
	};

	template<>
	struct mtimes_op<f32, 2, 3, 3>
	{
		LSIMD_ENSURE_INLINE
		static void run(
				const smat_core<f32, 2, 3>& A,
				const smat_core<f32, 3, 3>& B,
				      smat_core<f32, 2, 3>& C)
		{
			sse_f32pk ac0 = A.col01_pk.dup_low();
			sse_f32pk ac1 = A.col01_pk.dup_high();
			sse_f32pk ac2 = A.col2z_pk.dup_low();

			sse_f32pk bc0 = B.col0.m_pk;
			sse_f32pk bc1 = B.col1.m_pk;
			sse_f32pk bc2 = B.col2.m_pk;
			sse_f32pk z = zero_t();

			C.col01_pk =
					(ac0 * shuffle<0,0,0,0>(bc0, bc1)) +
					(ac1 * shuffle<1,1,1,1>(bc0, bc1)) +
					(ac2 * shuffle<2,2,2,2>(bc0, bc1));

			C.col2z_pk =
					(ac0 * shuffle<0,0,0,0>(bc2, z)) +
					(ac1 * shuffle<1,1,1,1>(bc2, z)) +
					(ac2 * shuffle<2,2,2,2>(bc2, z));
		}
	};


	template<>
	struct mtimes_op<f32, 2, 3, 4>
	{
		LSIMD_ENSURE_INLINE
		static void run(
				const smat_core<f32, 2, 3>& A,
				const smat_core<f32, 3, 4>& B,
				      smat_core<f32, 2, 4>& C)
		{
			sse_f32pk ac0 = A.col01_pk.dup_low();
			sse_f32pk ac1 = A.col01_pk.dup_high();
			sse_f32pk ac2 = A.col2z_pk.dup_low();

			sse_f32pk bc0 = B.col0.m_pk;
			sse_f32pk bc1 = B.col1.m_pk;
			sse_f32pk bc2 = B.col2.m_pk;
			sse_f32pk bc3 = B.col3.m_pk;

			C.col01_pk =
					(ac0 * shuffle<0,0,0,0>(bc0, bc1)) +
					(ac1 * shuffle<1,1,1,1>(bc0, bc1)) +
					(ac2 * shuffle<2,2,2,2>(bc0, bc1));

			C.col23_pk =
					(ac0 * shuffle<0,0,0,0>(bc2, bc3)) +
					(ac1 * shuffle<1,1,1,1>(bc2, bc3)) +
					(ac2 * shuffle<2,2,2,2>(bc2, bc3));
		}
	};



	/******************************************************
	 *
	 *  Specialized implementation for (2x4) * (4xN)
	 *
	 ******************************************************/

	template<>
	struct mtimes_op<f32, 2, 4, 2>
	{
		LSIMD_ENSURE_INLINE
		static void run(
				const smat_core<f32, 2, 4>& A,
				const smat_core<f32, 4, 2>& B,
				      smat_core<f32, 2, 2>& C)
		{
			sse_f32pk ac0 = A.col01_pk.dup_low();
			sse_f32pk ac1 = A.col01_pk.dup_high();
			sse_f32pk ac2 = A.col23_pk.dup_low();
			sse_f32pk ac3 = A.col23_pk.dup_high();

			sse_f32pk bc0 = B.col0.m_pk;
			sse_f32pk bc1 = B.col1.m_pk;

			sse_f32pk p11 = (ac0 * shuffle<0,0,0,0>(bc0, bc1)) +
							(ac1 * shuffle<1,1,1,1>(bc0, bc1));

			sse_f32pk p12 = (ac2 * shuffle<2,2,2,2>(bc0, bc1)) +
							(ac3 * shuffle<3,3,3,3>(bc0, bc1));

			C.col01_pk = p11 + p12;
		}
	};

	template<>
	struct mtimes_op<f32, 2, 4, 3>
	{
		LSIMD_ENSURE_INLINE
		static void run(
				const smat_core<f32, 2, 4>& A,
				const smat_core<f32, 4, 3>& B,
				      smat_core<f32, 2, 3>& C)
		{
			sse_f32pk ac0 = A.col01_pk.dup_low();
			sse_f32pk ac1 = A.col01_pk.dup_high();
			sse_f32pk ac2 = A.col23_pk.dup_low();
			sse_f32pk ac3 = A.col23_pk.dup_high();

			sse_f32pk bc0 = B.col0.m_pk;
			sse_f32pk bc1 = B.col1.m_pk;
			sse_f32pk bc2 = B.col2.m_pk;
			sse_f32pk z = zero_t();

			sse_f32pk p11 = (ac0 * shuffle<0,0,0,0>(bc0, bc1)) +
							(ac1 * shuffle<1,1,1,1>(bc0, bc1));

			sse_f32pk p12 = (ac2 * shuffle<2,2,2,2>(bc0, bc1)) +
							(ac3 * shuffle<3,3,3,3>(bc0, bc1));

			sse_f32pk p21 = (ac0 * shuffle<0,0,0,0>(bc2, z)) +
							(ac1 * shuffle<1,1,1,1>(bc2, z));

			sse_f32pk p22 = (ac2 * shuffle<2,2,2,2>(bc2, z)) +
							(ac3 * shuffle<3,3,3,3>(bc2, z));

			C.col01_pk = p11 + p12;
			C.col2z_pk = p21 + p22;
		}
	};


	template<>
	struct mtimes_op<f32, 2, 4, 4>
	{
		LSIMD_ENSURE_INLINE
		static void run(
				const smat_core<f32, 2, 4>& A,
				const smat_core<f32, 4, 4>& B,
				      smat_core<f32, 2, 4>& C)
		{
			sse_f32pk ac0 = A.col01_pk.dup_low();
			sse_f32pk ac1 = A.col01_pk.dup_high();
			sse_f32pk ac2 = A.col23_pk.dup_low();
			sse_f32pk ac3 = A.col23_pk.dup_high();

			sse_f32pk bc0 = B.col0.m_pk;
			sse_f32pk bc1 = B.col1.m_pk;
			sse_f32pk bc2 = B.col2.m_pk;
			sse_f32pk bc3 = B.col3.m_pk;

			sse_f32pk p11 = (ac0 * shuffle<0,0,0,0>(bc0, bc1)) +
							(ac1 * shuffle<1,1,1,1>(bc0, bc1));

			sse_f32pk p12 = (ac2 * shuffle<2,2,2,2>(bc0, bc1)) +
							(ac3 * shuffle<3,3,3,3>(bc0, bc1));

			sse_f32pk p21 = (ac0 * shuffle<0,0,0,0>(bc2, bc3)) +
							(ac1 * shuffle<1,1,1,1>(bc2, bc3));

			sse_f32pk p22 = (ac2 * shuffle<2,2,2,2>(bc2, bc3)) +
							(ac3 * shuffle<3,3,3,3>(bc2, bc3));

			C.col01_pk = p11 + p12;
			C.col23_pk = p21 + p22;
		}
	};


	/******************************************************
	 *
	 *  Specialized implementation for (3x2) * (2xN)
	 *
	 ******************************************************/

	template<>
	struct mtimes_op<f32, 3, 2, 2>
	{
		LSIMD_ENSURE_INLINE
		static void run(
				const smat_core<f32, 3, 2>& A,
				const smat_core<f32, 2, 2>& B,
				      smat_core<f32, 3, 2>& C)
		{
			C.col0 = A.col0 * B.col01_pk.bsx<0>() + A.col1 * B.col01_pk.bsx<1>();
			C.col1 = A.col0 * B.col01_pk.bsx<2>() + A.col1 * B.col01_pk.bsx<3>();
		}
	};

	template<>
	struct mtimes_op<f32, 3, 2, 3>
	{
		LSIMD_ENSURE_INLINE
		static void run(
				const smat_core<f32, 3, 2>& A,
				const smat_core<f32, 2, 3>& B,
				      smat_core<f32, 3, 3>& C)
		{
			C.col0 = A.col0 * B.col01_pk.bsx<0>() + A.col1 * B.col01_pk.bsx<1>();
			C.col1 = A.col0 * B.col01_pk.bsx<2>() + A.col1 * B.col01_pk.bsx<3>();
			C.col2 = A.col0 * B.col2z_pk.bsx<0>() + A.col1 * B.col2z_pk.bsx<1>();
		}
	};

	template<>
	struct mtimes_op<f32, 3, 2, 4>
	{
		LSIMD_ENSURE_INLINE
		static void run(
				const smat_core<f32, 3, 2>& A,
				const smat_core<f32, 2, 4>& B,
				      smat_core<f32, 3, 4>& C)
		{
			C.col0 = A.col0 * B.col01_pk.bsx<0>() + A.col1 * B.col01_pk.bsx<1>();
			C.col1 = A.col0 * B.col01_pk.bsx<2>() + A.col1 * B.col01_pk.bsx<3>();
			C.col2 = A.col0 * B.col23_pk.bsx<0>() + A.col1 * B.col23_pk.bsx<1>();
			C.col3 = A.col0 * B.col23_pk.bsx<2>() + A.col1 * B.col23_pk.bsx<3>();
		}
	};


	/******************************************************
	 *
	 *  Specialized implementation for (3x2) * (2xN)
	 *
	 ******************************************************/

	template<>
	struct mtimes_op<f32, 4, 2, 2>
	{
		LSIMD_ENSURE_INLINE
		static void run(
				const smat_core<f32, 4, 2>& A,
				const smat_core<f32, 2, 2>& B,
				      smat_core<f32, 4, 2>& C)
		{
			C.col0 = A.col0 * B.col01_pk.bsx<0>() + A.col1 * B.col01_pk.bsx<1>();
			C.col1 = A.col0 * B.col01_pk.bsx<2>() + A.col1 * B.col01_pk.bsx<3>();
		}
	};

	template<>
	struct mtimes_op<f32, 4, 2, 3>
	{
		LSIMD_ENSURE_INLINE
		static void run(
				const smat_core<f32, 4, 2>& A,
				const smat_core<f32, 2, 3>& B,
				      smat_core<f32, 4, 3>& C)
		{
			C.col0 = A.col0 * B.col01_pk.bsx<0>() + A.col1 * B.col01_pk.bsx<1>();
			C.col1 = A.col0 * B.col01_pk.bsx<2>() + A.col1 * B.col01_pk.bsx<3>();
			C.col2 = A.col0 * B.col2z_pk.bsx<0>() + A.col1 * B.col2z_pk.bsx<1>();
		}
	};

	template<>
	struct mtimes_op<f32, 4, 2, 4>
	{
		LSIMD_ENSURE_INLINE
		static void run(
				const smat_core<f32, 4, 2>& A,
				const smat_core<f32, 2, 4>& B,
				      smat_core<f32, 4, 4>& C)
		{
			C.col0 = A.col0 * B.col01_pk.bsx<0>() + A.col1 * B.col01_pk.bsx<1>();
			C.col1 = A.col0 * B.col01_pk.bsx<2>() + A.col1 * B.col01_pk.bsx<3>();
			C.col2 = A.col0 * B.col23_pk.bsx<0>() + A.col1 * B.col23_pk.bsx<1>();
			C.col3 = A.col0 * B.col23_pk.bsx<2>() + A.col1 * B.col23_pk.bsx<3>();
		}
	};


} }

#endif /* SSE_MAT_MATMUL_BITS_H_ */
