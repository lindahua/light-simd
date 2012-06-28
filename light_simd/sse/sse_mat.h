/**
 * @file sse_mat.h
 *
 * @brief SSE-based fixed-size small matrices
 *
 * @author Dahua Lin
 *
 * @copyright
 *
 * Copyright (C) 2012 Dahua Lin
 * 
 * Permission is hereby granted, free of charge, to any person 
 * obtaining a copy of this software and associated documentation 
 * files (the "Software"), to deal in the Software without restriction, 
 * including without limitation the rights to use, copy, modify, merge, 
 * publish, distribute, sublicense, and/or sell copies of the Software, 
 * and to permit persons to whom the Software is furnished to do so, 
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be 
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_MAT_H_
#define LSIMD_SSE_MAT_H_

#include "details/sse_mat_comp_bits.h"
#include "details/sse_mat_matmul_bits.h"
#include "details/sse_mat_sol_bits.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4141)
#endif


namespace lsimd
{

	/**
	 * \addtogroup mat_vec_sse
	 */
	/** @{ */ 


	template<typename T, int M, int N> class sse_mat;


	/**
	 * @brief SSE-based fixed size matrix.
	 *
	 * @tparam T    The entry value type.
	 * @tparam M    The number of rows.
	 * @tparam N    The number of columns.
	 *
	 * @remark      The entries are in column-major order.
	 */
	template<typename T, int M, int N>
	class sse_mat
	{
	public:
		sse::smat_core<T, M, N> core;

		LSIMD_ENSURE_INLINE
		sse_mat(const sse::smat_core<T, M, N>& a) : core(a) { }

	public:
		/**
		 * Default constructor.
		 *
		 * All entries of the matrix are left uninitialized.
		 */
		LSIMD_ENSURE_INLINE
		sse_mat() { }

		/**
		 * Constructs a matrix with all entries initialized to zeros.
		 */
		LSIMD_ENSURE_INLINE
		sse_mat( zero_t ) : core( zero_t() ) { }

		/**
		 * Constructs a matrix by loading entry values from an aligned
		 * memory address.
		 *
		 * @param x    The memory address from which the values are loaded.
		 */
		LSIMD_ENSURE_INLINE
		sse_mat(const T *x, aligned_t)
		{
			core.load(x, aligned_t());
		}

		/**
		 * Constructs a matrix by loading entry values from an address
		 * that is not necessarily aligned.
		 *
		 * @param x    The memory address from which the values are loaded.
		 */
		LSIMD_ENSURE_INLINE
		sse_mat(const T *x, unaligned_t)
		{
			core.load(x, unaligned_t());
		}

		/**
		 * Constructs a matrix by loading from strided memory, where each
		 * column is aligned.
		 *
		 * Specifically, the first column is loaded from x, the second is
		 * loaded from x + ldim, and so on. Here, the base addresses of
		 * all columns (x, x + ldim, etc) should be aligned.
		 *
		 * @param x    	The base address of the first column. 
		 * @param ldim 	The stride (the offset between two consecutive columns).
		 */
		LSIMD_ENSURE_INLINE
		sse_mat(const T *x, int ldim, aligned_t)
		{
			core.load(x, ldim, aligned_t());
		}

		/**
		 * Constructs a matrix by loading from strided memory, where the 
		 * column bases are not necessarily aligned.
		 *
		 * Specifically, the first column is loaded from x, the second is
		 * loaded from x + ldim, and so on. 
		 *
		 * @param x    	The base address of the first column. 
		 * @param ldim 	The stride (the offset between two consecutive columns).
		 */
		LSIMD_ENSURE_INLINE
		sse_mat(const T *x, int ldim, unaligned_t)
		{
			core.load(x, ldim, unaligned_t());
		}

		/**
		 * Loads entry values from an aligned memory address.
		 *
		 * @param x    The memory address from which the values are loaded.
		 */
		LSIMD_ENSURE_INLINE
		void load(const T *x, aligned_t)
		{
			core.load(x, aligned_t());
		}

		/**
		 * Loads entry values from an address that is not 
		 * necessarily aligned.
		 *
		 * @param x    The memory address from which the values are loaded.
		 */
		LSIMD_ENSURE_INLINE
		void load(const T *x, unaligned_t)
		{
			core.load(x, unaligned_t());
		}

		/**
		 * Loads from strided memory, where each column is aligned.
		 *
		 * Specifically, the first column is loaded from x, the second is
		 * loaded from x + ldim, and so on. Here, the base addresses of
		 * all columns (x, x + ldim, etc) should be aligned.
		 *
		 * @param x    	The base address of the first column. 
		 * @param ldim 	The stride (the offset between two consecutive columns).
		 */
		LSIMD_ENSURE_INLINE
		void load(const T *x, int ldim, aligned_t)
		{
			core.load(x, ldim, aligned_t());
		}

		/**
		 * Loads entry values from strided memory, where the base
		 * addresses are not necessarily aligned.
		 *
		 * Specifically, the first column is loaded from x, the
		 * second column is loaded from x + ldim, and so on.
		 *
		 * @param x     The base address of the first column.
		 * @param ldim  The stride (the offset between two consecutive columns).
		 */
		LSIMD_ENSURE_INLINE
		void load(const T *x, int ldim, unaligned_t)
		{
			core.load(x, ldim, unaligned_t());
		}

		/**
		 * Loads from an aligned memory address and transpose.
		 *
		 * Here, the matrix stored in the memory have M columns and
		 * each column has N entries.
		 *
		 * @param x    The memory address from which the values are loaded.
		 * 
		 * @remark     One can also use this function to load an M x N matrix
		 *             from a matrix in memory with row-major layout.
		 */
		LSIMD_ENSURE_INLINE
		void load_trans(const T *x, aligned_t)
		{
			core.load_trans(x, aligned_t());
		}

		/**
		 * Loads from a memory address (not necessarily aligned) and
		 * transpose.
		 *
		 * Here, the matrix stored in the memory have M columns and
		 * each column has N entries.
		 *
		 * @param x    The memory address from which the values are loaded.
		 * 
		 * @remark     One can also use this function to load an M x N matrix
		 *             from a matrix in memory with row-major layout.
		 */
		LSIMD_ENSURE_INLINE
		void load_trans(const T *x, unaligned_t)
		{
			core.load_trans(x, unaligned_t());
		}

		/**
		 * Loads from strided memory (where each column is aligned)
		 * and transpose.
		 *
		 * Here, the matrix stored in the memory have M columns and
		 * each column has N entries.
		 *
		 * @param x     The memory address from which the values are loaded.
		 * @param ldim  The stride (the offset between consecutive columns).
		 * 
		 * @remark     One can also use this function to load an M x N matrix
		 *             from a matrix in memory with row-major layout.
		 */
		LSIMD_ENSURE_INLINE
		void load_trans(const T *x, int ldim, aligned_t)
		{
			core.load_trans(x, ldim, aligned_t());
		}

		/**
		 * Loads from strided memory and transpose.
		 *
		 * Here, the matrix stored in the memory have M columns and
		 * each column has N entries.
		 *
		 * @param x     The memory address from which the values are loaded.
		 * @param ldim  The stride (the offset between consecutive columns).
		 * 
		 * @remark     One can also use this function to load an M x N matrix
		 *             from a matrix in memory with row-major layout.
		 */
		LSIMD_ENSURE_INLINE
		void load_trans(const T *x, int ldim, unaligned_t)
		{
			core.load_trans(x, ldim, unaligned_t());
		}

		/**
		 * Stores entry values to an aligned memory address.
		 *
		 * @param x    The memory address to which the entry values are stored.
		 */
		LSIMD_ENSURE_INLINE
		void store(T *x, aligned_t) const
		{
			core.store(x, aligned_t());
		}

		/**
		 * Stores entry values to a memory address that is not necessarily
		 * aligned.
		 *
		 * @param x    The memory address to which the entry values are stored.
		 */
		LSIMD_ENSURE_INLINE
		void store(T *x, unaligned_t) const
		{
			core.store(x, unaligned_t());
		}

		/**
		 * Stores entry values to strided memory, where each column is aligned.
		 *
		 * @param x     The memory address to which the entry values are stored.
		 * @param ldim  The stride (the offset between consecutive columns).
		 */
		LSIMD_ENSURE_INLINE
		void store(T *x, int ldim, aligned_t) const
		{
			core.store(x, ldim, aligned_t());
		}

		/**
		 * Stores entry values to strided memory (where the column base addresses
		 * are not required to be aligned).
		 *
		 * @param x     The memory address to which the entry values are stored.
		 * @param ldim  The stride (the offset between consecutive columns).
		 */
		LSIMD_ENSURE_INLINE
		void store(T *x, int ldim, unaligned_t) const
		{
			core.store(x, ldim, unaligned_t());
		}

	public:

		/**
		 * Adds two matrices.
		 *
		 * @param r    The matrix of addends.
		 * 
		 * @return     The resultant matrix, as this matrix + r.
		 */
		LSIMD_ENSURE_INLINE
		sse_mat operator + (const sse_mat& r) const
		{
			return core + r.core;
		}

		/**
		 * Subtracts two matrices.
		 *
		 * @param r    The matrix of subtrahends.
		 * 
		 * @return     The resultant matrix, as this matrix - r.
		 */
		LSIMD_ENSURE_INLINE
		sse_mat operator - (const sse_mat& r) const
		{
			return core - r.core;
		}

		/**
		 * Multiplies two matrices in an entry-wise way.
		 *
		 * @param r    The matrix of multiplicands.
		 * 
		 * @return     The resultant matrix, as an element-wise
		 *			   product between this matrix and r.
		 */
		LSIMD_ENSURE_INLINE
		sse_mat operator % (const sse_mat& r) const
		{
			return core % r.core;
		}

		/**
		 * Multiplies with a scale
		 *
		 * @param s    An SSE pack filled with the same scale.
		 * 
		 * @return     The resultant matrix, as this matrix * s.
		 */
		LSIMD_ENSURE_INLINE
		sse_mat operator * (const sse_pack<T>& s) const
		{
			return core * s;
		}

		/**
		 * Adds another matrix to this matrix.
		 *
		 * @param r    The matrix of addends.
		 * 
		 * @return     The reference to this matrix.
		 */
		LSIMD_ENSURE_INLINE
		sse_mat& operator += (const sse_mat& r)
		{
			core += r.core;
			return *this;
		}

		/**
		 * Subtracts another matrix from this matrix.
		 *
		 * @param r    The matrix of addends.
		 * 
		 * @return     The reference to this matrix.
		 */
		LSIMD_ENSURE_INLINE
		sse_mat& operator -= (const sse_mat& r)
		{
			core -= r.core;
			return *this;
		}

		/**
		 * Multiplies another matrix to this matrix, in an entry-wise way.
		 *
		 * @param r    The matrix of multiplicands.
		 * 
		 * @return     The reference to this matrix.
		 */
		LSIMD_ENSURE_INLINE
		sse_mat& operator %= (const sse_mat& r)
		{
			core %= r.core;
			return *this;
		}

		/**
		 * Multiplies this matrix with a scalar. 
		 *
		 * @param s   An SSE pack filled with the same scale value.
		 *
		 * @return    The reference to this matrix.
		 */
		LSIMD_ENSURE_INLINE
		sse_mat& operator *= (const sse_pack<T>& s)
		{
			core *= s;
			return *this;
		}

		/**
		 * Evaluates matrix-vector product (in linear algebraic sense).
		 *
		 * @param v   The vector to be multiplied with.
		 *
		 * @return    The matrix-vector product, as this matrix * v.
		 */
		LSIMD_ENSURE_INLINE
		sse_vec<T, M> operator * (const sse_vec<T, N>& v) const
		{
			return transform(core, v);
		}

		/**
		 * Evaluates the trace of the matrix.
		 *
		 * @return   The trace value.
		 *
		 * @remarks  The trace of a matrix equals the sum of all diagonal
		 *           entries.
		 */
		LSIMD_ENSURE_INLINE
		T trace() const
		{
			return core.trace();
		}

	public:
		LSIMD_ENSURE_INLINE
		bool test_equal(const T *r) const
		{
			return core.test_equal(r);
		}

		LSIMD_ENSURE_INLINE
		void dump(const char *fmt) const
		{
			core.dump(fmt);
		}

	};

	/**
	 * Evaluates matrix-matrix product.
	 *
	 * @param A    A matrix of size M x K.
	 * @param B    A matrix of size K x N.
	 *
	 * @return     The product of A and B, whose size is M x N.
	 */ 
	template<typename T, int M, int K, int N>
	inline sse_mat<T, M, N> operator * (const sse_mat<T, M, K>& A, const sse_mat<T, K, N>& B)
	{
		sse_mat<T, M, N> C;
		sse::mtimes_op<T, M, K, N>::run(A.core, B.core, C.core);
		return C;
	}

	/** @} */  // mat_vec_sse


	/**
	 * \defgroup linsol_sse SSE Equation Solving
	 * @ingroup linalg_module
	 * 
	 * @brief SSE-based functions to solve linear equations, 
	 *		  evaluate matrix determinant and inverse.
	 */
	 /** @{ */

	/**
	 * Evaluate the determinant of a matrix.
	 *
	 * @param A   The input matrix.
	 *
	 * @return    The determinant of A.
	 */
	template<typename T, int N>
	LSIMD_ENSURE_INLINE
	inline T det(const sse_mat<T, N, N>& A)
	{
		return sse::det(A.core);
	}

	/**
	 * Evaluates the inverse of a matrix.
	 *
	 * @param A    The input matrix.
	 *
	 * @return     The inverse of A.
	 */
	template<typename T, int N>
	inline sse_mat<T, N, N> inv(const sse_mat<T, N, N>& A)
	{
		sse_mat<T, N, N> R;
		sse::inv(A.core, R.core);
		return R;
	}

	/**
	 * Evaluates the inverse and determinant of a matrix.
	 *
	 * @param A    The input matrix.
	 * @param R    The output matrix that stores the inverse of A.
	 *
	 * @return     The determinant of A.
	 */
	template<typename T, int N>
	inline T inv_and_det(const sse_mat<T, N, N>& A, sse_mat<T, N, N>& R)
	{
		return sse::inv(A.core, R.core);
	}

	/**
	 * Solves a linear equation.
	 *
	 * @param A    The input matrix of equation coefficients.
	 * @param b    The right hand side vector.
	 *
	 * @return     The solution x, such that A * x = b.
	 *
	 * @remark     This function assumes A is invertible.
	 */
	template<typename T, int N>
	inline sse_vec<T, N> solve(const sse_mat<T, N, N>& A, const sse_vec<T, N>& b)
	{
		sse_vec<T, N> x;
		sse::solve(A.core, b, x);
		return x;
	}

	/**
	 * Solves linear equations.
	 *
	 * @param A    The input matrix of equation coefficients.
	 * @param B    The right hand side matrix.
	 *
	 * @return     The solution matrix X, such that A * X = B.
	 *
	 * @remark     This function assumes A is invertible.
	 */
	template<typename T, int N, int N2>
	inline sse_mat<T, N, N2> solve(const sse_mat<T, N, N>& A, const sse_mat<T, N, N2>& B)
	{
		sse_mat<T, N, N2> X;
		sse::solve(A.core, B.core, X.core);
		return X;
	}

	/** @} */

}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif 
