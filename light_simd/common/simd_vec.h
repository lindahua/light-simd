/**
 * @file simd_vec.h
 *
 * @brief SIMD-based fixed-size vector classes.
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

#ifndef LSIMD_SIMD_VEC_H_
#define LSIMD_SIMD_VEC_H_

#include "simd_pack.h"
#include <light_simd/sse/sse_vec.h>

namespace lsimd
{
	/**
	 * @defgroup linalg_module Linear Algebra Module
	 *
	 * @brief Fixed-size vector and matrix classes and linear algebraic computation.
	 *
	 * This module comprises several classes to represent fixed size vectors 
	 * and matrices, and a set of functions to support linear algebraic computation.
	 *
	 * Particularly, it contains
	 * - generic fixed-size vector and matrix classes;
	 * - specialized vector and matrix classes based on specific architectures; 
	 * - linear algebraic functions that act on such classes, such as
	 *   matrix multiplication, inversion, and equation solving.
	 */

	/**
	 * @defgroup mat_vec_generic Generic Vectors and Matrices
	 * @ingroup  linalg_module
	 *
	 * @brief Generic fixed-size vector and matrix classes.
	 *
	 * These are template classes that wrap the architecture-specific vector
	 * and matrix classes to support generic programming.  
	 */
	/** @{ */ 

	template<typename T, int N, typename Kind>
	struct simd_vec_traits;

	template<typename T, int N>
	struct simd_vec_traits<T, N, sse_kind>
	{
		typedef sse_vec<T, N> impl_type;
	};


	/**
	 * @brief Generic fixed-size vector class.
	 *
	 * @tparam T 		The scalar type.
	 * @tparam N 		The vector length.
	 * @tparam Kind 	The kind of architecture.
	 */
	template<typename T, int N, typename Kind>
	struct simd_vec
	{
		/**
		 * The scalar type.
		 */
		typedef T value_type;

		/**
		 * The underlying architecture-specific vector type.
		 */
		typedef typename simd_vec_traits<T, N, Kind>::impl_type impl_type;

		/**
		 * The corresponding SIMD pack type.
		 */
		typedef simd_pack<T, Kind> pack_type;

		/**
		 * The (architecture-specific) internal vector that actually
		 * implements the functionality.
		 */
		impl_type impl;

		/**
		 * Default constructor.
		 *
		 * Constructs a vector with the entry values left uninitialized.
		 */
		LSIMD_ENSURE_INLINE simd_vec() { }

		/**
		 * Constructs a vector with all entries initialized to be zeros.
		 *
		 * @post this vector == (0, 0, ..., 0).
		 */
		LSIMD_ENSURE_INLINE simd_vec( zero_t ) : impl( zero_t() ) { }

		/**
		 * Constructs a vector using the architecture-specific vector
		 *
		 * @param imp   The implementing vector object.
		 */
		LSIMD_ENSURE_INLINE simd_vec( const impl_type& imp ) : impl(imp) { }

		/**
		 * Constructs a vector by loading from an aligned memory address
		 *
		 * @param x   The memory address from which values are loaded.
		 * 
		 * @post this vector == (x[0], ..., x[N-1]).
		 */
		LSIMD_ENSURE_INLINE simd_vec( const T *x, aligned_t )
		: impl(x, aligned_t()) { }

		/**
		 * Constructs a vector by loading from a memory address that
		 * is not necessarily aligned.
		 *
		 * @param x   The memory address from which values are loaded.
		 * 
		 * @post this vector == (x[0], ..., x[N-1]).
		 */
		LSIMD_ENSURE_INLINE simd_vec( const T *x, unaligned_t )
		: impl(x, unaligned_t()) { }

		
		/**
		 * Loads entry values from an aligned memory address.
		 *
		 * @param x   The memory address from which the values are loaded.
		 * 
		 * @post  this vector = (x[0], ..., x[N-1]).
		 */
		LSIMD_ENSURE_INLINE void load( const T *x, aligned_t )
		{
			impl.load(x, aligned_t() );
		}

		/**
		 * Loads entry values from a memory address that is not
		 * necessarily aligned.
		 *
		 * @param x   The memory address from which the values are loaded.
		 *
		 * @post  this vector = (x[0], ..., x[N-1]).
		 */
		LSIMD_ENSURE_INLINE void load( const T *x, unaligned_t )
		{
			impl.load(x, unaligned_t() );
		}

		/**
		 * Stores entry values to an aligned memory address.
		 *
		 * @param x   The memory address to which the values are stored.
		 *
		 * @post  (x[0], ..., x[N-1]) = this vector.
		 */
		LSIMD_ENSURE_INLINE void store( T *x, aligned_t ) const
		{
			impl.store(x, aligned_t() );
		}

		/**
		 * Stores entry values to a memory address that is not necessarily
		 * aligned.
		 *
		 * @param x   The memory address to which the values are stored.
		 *
		 * @post  (x[0], ..., x[N-1]) = this vector.
		 */
		LSIMD_ENSURE_INLINE void store( T *x, unaligned_t ) const
		{
			impl.store(x, unaligned_t() );
		}

		/**
		 * Adds two SIMD vectors.
		 *
		 * @param    rhs   The vector of addends.
		 *
		 * @return   The resultant vector.
		 */
		LSIMD_ENSURE_INLINE simd_vec operator + (const simd_vec& rhs) const
		{
			return impl + rhs.impl;
		}

		/**
		 * Subtracts two SIMD vectors.
		 *
		 * @param   rhs   The vector of subtrahends.
		 *
		 * @return  The resultant vector. 
		 */
		LSIMD_ENSURE_INLINE simd_vec operator - (const simd_vec& rhs) const
		{
			return impl - rhs.impl;
		}

		/**
		 * Multiplies two SIMD vectors in an entry-wise way.
		 *
		 * @param   rhs   The vector of multiplicands.
		 *
		 * @return  The resultant vector. 
		 */
		LSIMD_ENSURE_INLINE simd_vec operator % (const simd_vec& rhs) const
		{
			return impl % rhs.impl;
		}

		/**
		 * Adds another SIMD vector to this vector.
		 *
		 * @param    rhs  The SIMD vector to be added.
		 * 
		 * @return   The reference to this vector.
		 */
		LSIMD_ENSURE_INLINE simd_vec& operator += (const simd_vec& rhs)
		{
			impl += rhs.impl;
			return *this;
		}

		/**
		 * Subtracts another SIMD vector from this vector.
		 *
		 * @param    rhs  The SIMD vector to be subtracted.
		 * 
		 * @return   The reference to this vector.
		 */
		LSIMD_ENSURE_INLINE simd_vec& operator -= (const simd_vec& rhs)
		{
			impl -= rhs.impl;
			return *this;
		}

		/**
		 * Multiplies another SIMD vector to this vector.
		 *
		 * @param    rhs  The SIMD vector to be multiplied.
		 * 
		 * @return   The reference to this vector.
		 */
		LSIMD_ENSURE_INLINE simd_vec& operator %= (const simd_vec& rhs)
		{
			impl %= rhs.impl;
			return *this;
		}

		/**
		 * Computes scalar product.
		 *
		 * @param   s  An SIMD pack filled with the same scale value.
		 *
		 * @return  The resultant vector.
		 */
		LSIMD_ENSURE_INLINE simd_vec operator * (const pack_type& s) const
		{
			return impl * s.impl;
		}

		/**
		 * Multiplies this vector with a scale.
		 *
		 * @param   s  An SIMD pack filled with the same scale value.
		 *
		 * @return  The reference to this vector.
		 */
		LSIMD_ENSURE_INLINE simd_vec& operator *= (const pack_type& s)
		{
			impl *= s.impl;
			return *this;
		}

		// stats

		/**
		 * Computes the sum of all entries.
		 *
		 * @return  The sum of all entries.
		 */
		LSIMD_ENSURE_INLINE T sum() const
		{
			return impl.sum();
		}

		/**
		 * Computes the dot product with another SIMD vector.
		 *
		 * @param   rhs  Another SIMD vector.
		 *
		 * @return  The dot product value.
		 */
		LSIMD_ENSURE_INLINE T dot(const simd_vec& rhs) const
		{
			return impl.dot(rhs.impl);
		}

	};

	/** @} */

}

#endif /* SIMD_MATMUL_H_ */
