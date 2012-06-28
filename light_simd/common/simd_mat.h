/**
 * @file simd_mat.h
 *
 * @brief SIMD-based fixed-size matrix classes
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

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SIMD_MAT_H_
#define LSIMD_SIMD_MAT_H_

#include "simd_vec.h"
#include <light_simd/sse/sse_mat.h>

namespace lsimd
{

	/**
	 * \addtogroup mat_vec_generic
	 */
	/** @{ */ 

	template<typename T, int M, int N, typename Kind>
	struct simd_mat_traits;

	template<typename T, int M, int N>
	struct simd_mat_traits<T, M, N, sse_kind>
	{
		typedef sse_mat<T, M, N> impl_type;
	};

	/**
	 * @brief Generic fixed size matrix.
	 *  
	 * @tparam T    The entry value type.
	 * @tparam M    The number of rows.
	 * @tparam N    The number of columns.
	 *
	 * @remarks     The entries are in column-major order.
	 */
	template<typename T, int M, int N, typename Kind>
	struct simd_mat
	{
		/**
		 * The entry value type.
		 */ 
		typedef T value_type;

		/**
		 * The architecture-specific type that provides the internal
		 * implementation.
		 */
		typedef typename simd_mat_traits<T, M, N, Kind>::impl_type impl_type;

		/**
		 * The corresponding SIMD pack type.
		 */
		typedef simd_pack<T, Kind> pack_type;
		
		/**
		 * The variable that actually implements the functionalities.
		 */
		impl_type impl;


		/**
		 * Default constructor.
		 *
		 * All entries of the matrix are left uninitialized.
		 */
		LSIMD_ENSURE_INLINE
		simd_mat() { }

		/**
		 * Constructs a matrix with all entries initialized to zeros.
		 */
		LSIMD_ENSURE_INLINE
		simd_mat( zero_t ) : impl( zero_t() ) { }

		/**
		 * Constructs a matrix using the internal implementation.
		 *
		 * @param imp    The internal implementation.
		 */
		LSIMD_ENSURE_INLINE
		simd_mat( const impl_type& imp ) : impl(imp) { }

		/**
		 * Constructs a matrix by loading entry values from an aligned
		 * memory address.
		 *
		 * @param x    The memory address from which the values are loaded.
		 */
		LSIMD_ENSURE_INLINE
		simd_mat(const T *x, aligned_t)
		: impl(x, aligned_t()) { }

		/**
		 * Constructs a matrix by loading entry values from an address
		 * that is not necessarily aligned.
		 *
		 * @param x    The memory address from which the values are loaded.
		 */
		LSIMD_ENSURE_INLINE
		simd_mat(const T *x, unaligned_t)
		: impl(x, unaligned_t()) { }

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
		simd_mat(const T *x, int ldim, aligned_t)
		: impl(x, ldim, aligned_t()) { }

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
		simd_mat(const T *x, int ldim, unaligned_t)
		: impl(x, ldim, unaligned_t()) { }


		/**
		 * Loads entry values from an aligned memory address.
		 *
		 * @param x    The memory address from which the values are loaded.
		 */
		LSIMD_ENSURE_INLINE
		void load(const T *x, aligned_t)
		{
			impl.load(x, aligned_t());
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
			impl.load(x, unaligned_t());
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
			impl.load(x, ldim, aligned_t());
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
			impl.load(x, ldim, unaligned_t());
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
			impl.load_trans(x, aligned_t());
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
			impl.load_trans(x, unaligned_t());
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
			impl.load_trans(x, ldim, aligned_t());
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
			impl.load_trans(x, ldim, unaligned_t());
		}

		/**
		 * Stores entry values to an aligned memory address.
		 *
		 * @param x    The memory address to which the entry values are stored.
		 */
		LSIMD_ENSURE_INLINE
		void store(T *x, aligned_t) const
		{
			impl.store(x, aligned_t());
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
			impl.store(x, unaligned_t());
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
			impl.store(x, ldim, aligned_t());
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
			impl.store(x, ldim, unaligned_t());
		}


		/**
		 * Adds two matrices.
		 *
		 * @param r    The matrix of addends.
		 * 
		 * @return     The resultant matrix, as this matrix + r.
		 */
		LSIMD_ENSURE_INLINE
		simd_mat operator + (const simd_mat& r) const
		{
			return impl + r.impl;
		}

		/**
		 * Subtracts two matrices.
		 *
		 * @param r    The matrix of subtrahends.
		 * 
		 * @return     The resultant matrix, as this matrix - r.
		 */
		LSIMD_ENSURE_INLINE
		simd_mat operator - (const simd_mat& r) const
		{
			return impl - r.impl;
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
		simd_mat operator % (const simd_mat& r) const
		{
			return impl % r.impl;
		}

		/**
		 * Multiplies with a scale
		 *
		 * @param s    An SSE pack filled with the same scale.
		 * 
		 * @return     The resultant matrix, as this matrix * s.
		 */
		LSIMD_ENSURE_INLINE
		simd_mat operator * (const pack_type& s) const
		{
			return impl * s.impl;
		}

		/**
		 * Adds another matrix to this matrix.
		 *
		 * @param r    The matrix of addends.
		 * 
		 * @return     The reference to this matrix.
		 */
		LSIMD_ENSURE_INLINE
		simd_mat& operator += (const simd_mat& r)
		{
			impl += r.impl;
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
		simd_mat& operator -= (const simd_mat& r)
		{
			impl -= r.impl;
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
		simd_mat& operator %= (const simd_mat& r)
		{
			impl %= r.impl;
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
		simd_mat& operator *= (const pack_type& s)
		{
			impl *= s.impl;
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
		simd_vec<T, M> operator * (const simd_vec<T, N>& v) const
		{
			return impl * v.impl;
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
			return impl.trace();
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
	template<typename Kind, typename T, int M, int K, int N>
	inline simd_mat<T, M, N, Kind> operator * (
			const simd_mat<T, M, K, Kind>& A,
			const simd_mat<T, K, N, Kind>& B)
	{
		simd_mat<T, M, N, Kind> C;
		C.impl = A.impl * B.impl;
		return C;
	}

	/** @} */ // mat_vec_generic


	/**
	 * \defgroup linsol_generic Generic Equation Solving
	 * @ingroup linalg_module
	 * 
	 * @brief Generic functions to solve linear equations, 
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
	template<typename Kind, typename T, int N>
	inline T det(const simd_mat<T, N, N, Kind>& A)
	{
		return det(A.impl);
	}

	/**
	 * Evaluates the inverse of a matrix.
	 *
	 * @param A    The input matrix.
	 *
	 * @return     The inverse of A.
	 */
	template<typename Kind, typename T, int N>
	inline simd_mat<T, N, N, Kind> inv(const simd_mat<T, N, N, Kind>& A)
	{
		return inv(A.impl);
	}

	/**
	 * Evaluates the inverse and determinant of a matrix.
	 *
	 * @param A    The input matrix.
	 * @param R    The output matrix that stores the inverse of A.
	 *
	 * @return     The determinant of A.
	 */
	template<typename Kind, typename T, int N>
	inline T inv_and_det(const simd_mat<T, N, N, Kind>& A, simd_mat<T, N, N, Kind>& R)
	{
		return inv_and_det(A.impl, R.impl);
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
	template<typename Kind, typename T, int N>
	inline simd_vec<T, N, Kind> solve(const simd_mat<T, N, N, Kind>& A, simd_vec<T, N, Kind> b)
	{
		return solve(A.impl, b.impl);
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
	template<typename Kind, typename T, int N, int N2>
	inline simd_mat<T, N, N2, Kind> solve(const simd_mat<T, N, N, Kind>& A, const simd_mat<T, N, N2, Kind>& B)
	{
		return solve(A.impl, B.impl);
	}

}

#endif /* SIMD_MAT_H_ */
