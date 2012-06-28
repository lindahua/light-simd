/**
 * @file sse_vec.h
 *
 * @brief SSE-based fixed-size vector classes.
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

#ifndef LSIMD_SSE_VEC_H_
#define LSIMD_SSE_VEC_H_

#include "sse_arith.h"

namespace lsimd
{

	/**
	 * @defgroup mat_vec_sse SSE Vectors and Matrices
	 * @ingroup  linalg_module
	 *
	 * @brief SSE-based fixed-size vector and matrix classes. 
	 */
	/** @{ */ 

	template<typename T, int N> class sse_vec;

#ifdef LSIMD_IN_DOXYGEN

	/**
	 * @brief A fixed-size SSE-based vector.
	 *
	 * @tparam T   The type of entry values.
	 * @tparam N   The number of entries in a vector.
	 */
	template<typename T, int N>
	class sse_vec
	{
	public:
		/**
		 * Default constructor.
		 *
		 * Constructs an SIMD vector with entries left uninitialized.
		 */
		LSIMD_ENSURE_INLINE sse_vec();

		/**
		 * Constructs an SIMD vector with all entries initialized to zeros.
		 *
		 * @post  this vector = (0, ..., 0).
		 */
		LSIMD_ENSURE_INLINE sse_vec( zero_t );

		/**
		 * Constructs an SIMD vector with all entries 
		 * initialized to a given value.
		 *
		 * @param e0  The value to be set to all entries.
		 *
		 * @post  this vector = (e0, ..., e0).
		 */
		LSIMD_ENSURE_INLINE sse_vec(const f32 e0);

		/**
		 * Constructs an SIMD vector by loading from an aligned memory address.
		 *
		 * @param x   The memory address from which the values are loaded.
		 *
		 * @post  this vector = (x[0], ..., x[N-1]).
		 */
		LSIMD_ENSURE_INLINE sse_vec(const f32 *x, aligned_t);

		/**
		 * Constructs an SIMD vector by loading from a memory address
		 * which is not necessarily aligned.
		 *
		 * @param x   The memory address from which the values are loaded.
		 *
		 * @post  this vector = (x[0], ..., x[N-1]).
		 */
		LSIMD_ENSURE_INLINE sse_vec(const f32 *x, unaligned_t);

		/**
		 * Loads the entry values from an aligned memory address.
		 * 
		 * @param x   The memory address from which the values are loaded.
		 *
		 * @post  this vector = (x[0], ..., x[N-1]).
		 */
		LSIMD_ENSURE_INLINE void load(const f32 *x, aligned_t);

		/**
		 * Loads the entry values from a memory address that is not
		 * necessarily aligned. 
		 * 
		 * @param x   The memory address from which the values are loaded.
		 *
		 * @post  this vector = (x[0], ..., x[N-1]).
		 */
		LSIMD_ENSURE_INLINE void load(const f32 *x, unaligned_t);

		/**
		 * Stores the entry values to an aligned memory address.
		 *
		 * @param x   The memory address to which the values are stored.
		 *
		 * @post  x[0:N-1] = this vector.
		 */
		LSIMD_ENSURE_INLINE void store(f32 *x, aligned_t) const;

		/**
		 * Stores the entry values to a memory address that is not 
		 * necessarily aligned.
		 *
		 * @param x   The memory address to which the values are stored.
		 *
		 * @post  x[0:N-1] = this vector.
		 */
		LSIMD_ENSURE_INLINE void store(f32 *x, unaligned_t) const;

		/**
		 * Generates an SSE pack of which all entries equal to 
		 * a particular entry of the vector.
		 *
		 * @tparam I   The index of the value to be used.
		 *
		 * @return     An SSE pack as (v[I], ..., v[I]), where v
		 *             refers to this vector.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE sse_f32pk bsx_pk() const;

	public:

		/**
		 * Adds two SIMD vectors.
		 *
		 * @param rhs   The vector of addends.
		 *
		 * @return   The resultant vector, as this vector + rhs.
		 */
		LSIMD_ENSURE_INLINE sse_vec operator + (const sse_vec& rhs) const;

		/**
		 * Subtracts two SIMD vectors.
		 *
		 * @param rhs   The vector of subtrahends.
		 *
		 * @return   The resultant vector, as this vector - rhs.
		 */
		LSIMD_ENSURE_INLINE sse_vec operator - (const sse_vec& rhs) const;

		/**
		 * Multiplies two SIMD vectors, in an entry-wise way.
		 *
		 * @param rhs   The vector of multiplicands.
		 *
		 * @return   The resultant vector, as this vector * rhs.
		 */
		LSIMD_ENSURE_INLINE sse_vec operator % (const sse_vec& rhs) const;

		/**
		 * Adds another vector to this vector.
		 *
		 * @param rhs   The vector of addends.
		 *
		 * @return   The reference to this vector.
		 */
		LSIMD_ENSURE_INLINE sse_vec& operator += (const sse_vec& rhs);

		/**
		 * Subtracts another vector from this vector.
		 *
		 * @param rhs   The vector of subtrahends.
		 *
		 * @return   The reference to this vector.
		 */
		LSIMD_ENSURE_INLINE sse_vec& operator -= (const sse_vec& rhs);

		/**
		 * Multiplies another vector from this vector.
		 *
		 * @param rhs   The vector of multiplicands.
		 *
		 * @return   The reference to this vector.
		 */
		LSIMD_ENSURE_INLINE sse_vec& operator %= (const sse_vec& rhs);

		/**
		 * Computes a scale product.
		 *
		 * @param s   The SSE pack filled with the scale values.
		 *
		 * @return    The resultant vector, as this vector * s.
		 *
		 * @post  All entries in s should be the same. 
		 */
		LSIMD_ENSURE_INLINE sse_vec operator * (const sse_f32pk& s) const;

		/**
		 * Scales this vector.
		 *
		 * @param s   The SSE pack filled with the scale values.
		 *
		 * @return    The reference to this vector.
		 *
		 * @post  All entries in s should be the same. 
		 */
		LSIMD_ENSURE_INLINE sse_vec& operator *= (const sse_f32pk& s);

	public:

		/**
		 * Computes the sum of all entries.
		 *
		 * @return the sum of all entries.
		 */
		LSIMD_ENSURE_INLINE f32 sum() const;

		/**
		 * Computes the dot product with another vector.
		 *
		 * @param rhs   Another vector involved in the dot product.
		 *
		 * @return  the dot prodcut of this vector and rhs.
		 */
		LSIMD_ENSURE_INLINE f32 dot(const sse_vec& rhs) const;
	};


#endif

	/********************************************
	 *
	 *  sse_vec class for f32
	 *
	 ********************************************/

	template<> class sse_vec<f32, 1>
	{
	public:
		LSIMD_ENSURE_INLINE explicit sse_vec(const __m128 p) : m_pk(p) { }
		LSIMD_ENSURE_INLINE explicit sse_vec(const sse_f32pk& p) : m_pk(p) { }

	public:
		LSIMD_ENSURE_INLINE sse_vec() { }

		LSIMD_ENSURE_INLINE sse_vec( zero_t ) : m_pk( zero_t() ) { }

		LSIMD_ENSURE_INLINE sse_vec(const f32 e0)
		{
			m_pk.v = _mm_set_ss(e0);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f32 *x, aligned_t)
		{
			m_pk.partial_load<1>(x);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f32 *x, unaligned_t)
		{
			m_pk.partial_load<1>(x);
		}

		LSIMD_ENSURE_INLINE void load(const f32 *x, aligned_t)
		{
			m_pk.partial_load<1>(x);
		}

		LSIMD_ENSURE_INLINE void load(const f32 *x, unaligned_t)
		{
			m_pk.partial_load<1>(x);
		}

		LSIMD_ENSURE_INLINE void store(f32 *x, aligned_t) const
		{
			m_pk.partial_store<1>(x);
		}

		LSIMD_ENSURE_INLINE void store(f32 *x, unaligned_t) const
		{
			m_pk.partial_store<1>(x);
		}

		template<int I>
		LSIMD_ENSURE_INLINE sse_f32pk bsx_pk() const
		{
			return m_pk.bsx<I>();
		}

	public:

		LSIMD_ENSURE_INLINE sse_vec operator + (const sse_vec& rhs) const
		{
			return sse_vec(add_s(m_pk, rhs.m_pk));
		}

		LSIMD_ENSURE_INLINE sse_vec operator - (const sse_vec& rhs) const
		{
			return sse_vec(sub_s(m_pk, rhs.m_pk));
		}

		LSIMD_ENSURE_INLINE sse_vec operator % (const sse_vec& rhs) const
		{
			return sse_vec(mul_s(m_pk, rhs.m_pk));
		}

		LSIMD_ENSURE_INLINE sse_vec& operator += (const sse_vec& rhs)
		{
			m_pk = add_s(m_pk, rhs.m_pk);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator -= (const sse_vec& rhs)
		{
			m_pk = sub_s(m_pk, rhs.m_pk);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator %= (const sse_vec& rhs)
		{
			m_pk = mul_s(m_pk, rhs.m_pk);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec operator * (const sse_f32pk& s) const
		{
			return sse_vec(mul_s(m_pk, s));
		}

		LSIMD_ENSURE_INLINE sse_vec& operator *= (const sse_f32pk& s)
		{
			m_pk = mul_s(m_pk, s);
			return *this;
		}

	public:
		LSIMD_ENSURE_INLINE f32 sum() const
		{
			return m_pk.to_scalar();
		}

		LSIMD_ENSURE_INLINE f32 dot(const sse_vec& rhs) const
		{
			return _mm_cvtss_f32(_mm_mul_ss(m_pk.v, rhs.m_pk.v));
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return m_pk.test_equal(r[0], 0.f, 0.f, 0.f);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [1]:\n");
			std::printf("    m_pk = "); m_pk.dump(fmt); std::printf("\n");
		}

	public:
		sse_f32pk m_pk;
	};


	template<> class sse_vec<f32, 2>
	{
	public:
		LSIMD_ENSURE_INLINE explicit sse_vec(__m128 p) : m_pk(p) { }
		LSIMD_ENSURE_INLINE explicit sse_vec(const sse_f32pk& p) : m_pk(p) { }

	public:
		LSIMD_ENSURE_INLINE sse_vec() { }

		LSIMD_ENSURE_INLINE sse_vec( zero_t ) : m_pk( zero_t() ) { }

		LSIMD_ENSURE_INLINE sse_vec(const f32 e0, const f32 e1)
		{
			m_pk.v = _mm_setr_ps(e0, e1, 0.f, 0.f);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f32 *x, aligned_t)
		{
			m_pk.partial_load<2>(x);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f32 *x, unaligned_t)
		{
			m_pk.partial_load<2>(x);
		}

		LSIMD_ENSURE_INLINE void load(const f32 *x, aligned_t)
		{
			m_pk.partial_load<2>(x);
		}

		LSIMD_ENSURE_INLINE void load(const f32 *x, unaligned_t)
		{
			m_pk.partial_load<2>(x);
		}

		LSIMD_ENSURE_INLINE void store(f32 *x, aligned_t) const
		{
			m_pk.partial_store<2>(x);
		}

		LSIMD_ENSURE_INLINE void store(f32 *x, unaligned_t) const
		{
			m_pk.partial_store<2>(x);
		}

		template<int I>
		LSIMD_ENSURE_INLINE sse_f32pk bsx_pk() const
		{
			return m_pk.bsx<I>();
		}

	public:

		LSIMD_ENSURE_INLINE sse_vec operator + (const sse_vec& rhs) const
		{
			return sse_vec(m_pk + rhs.m_pk);
		}

		LSIMD_ENSURE_INLINE sse_vec operator - (const sse_vec& rhs) const
		{
			return sse_vec(m_pk - rhs.m_pk);
		}

		LSIMD_ENSURE_INLINE sse_vec operator % (const sse_vec& rhs) const
		{
			return sse_vec(m_pk * rhs.m_pk);
		}

		LSIMD_ENSURE_INLINE sse_vec& operator += (const sse_vec& rhs)
		{
			m_pk = m_pk + rhs.m_pk;
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator -= (const sse_vec& rhs)
		{
			m_pk = m_pk - rhs.m_pk;
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator %= (const sse_vec& rhs)
		{
			m_pk = m_pk * rhs.m_pk;
			return *this;
		}


		LSIMD_ENSURE_INLINE sse_vec operator * (const sse_f32pk& s) const
		{
			return sse_vec(m_pk * s);
		}

		LSIMD_ENSURE_INLINE sse_vec& operator *= (const sse_f32pk& s)
		{
			m_pk = m_pk * s;
			return *this;
		}

	public:
		LSIMD_ENSURE_INLINE f32 sum() const
		{
			return m_pk.partial_sum<2>();
		}

		LSIMD_ENSURE_INLINE f32 dot(const sse_vec& rhs) const
		{
			return (m_pk * rhs.m_pk).partial_sum<2>();
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return m_pk.test_equal(r[0], r[1], 0.f, 0.f);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [2]:\n");
			std::printf("    m_pk = "); m_pk.dump(fmt); std::printf("\n");
		}

	public:
		sse_f32pk m_pk;
	};


	template<> class sse_vec<f32, 3>
	{
	public:
		LSIMD_ENSURE_INLINE explicit sse_vec(const __m128 p) : m_pk(p) { }
		LSIMD_ENSURE_INLINE explicit sse_vec(const sse_f32pk& p) : m_pk(p) { }

	public:
		LSIMD_ENSURE_INLINE sse_vec() { }

		LSIMD_ENSURE_INLINE sse_vec( zero_t ) : m_pk( zero_t() ) { }

		LSIMD_ENSURE_INLINE sse_vec(const f32 e0, const f32 e1, const f32 e2)
		{
			m_pk.v = _mm_setr_ps(e0, e1, e2, 0.f);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f32 *x, aligned_t)
		{
			m_pk.partial_load<3>(x);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f32 *x, unaligned_t)
		{
			m_pk.partial_load<3>(x);
		}

		LSIMD_ENSURE_INLINE void load(const f32 *x, aligned_t)
		{
			m_pk.partial_load<3>(x);
		}

		LSIMD_ENSURE_INLINE void load(const f32 *x, unaligned_t)
		{
			m_pk.partial_load<3>(x);
		}

		LSIMD_ENSURE_INLINE void store(f32 *x, aligned_t) const
		{
			m_pk.partial_store<3>(x);
		}

		LSIMD_ENSURE_INLINE void store(f32 *x, unaligned_t) const
		{
			m_pk.partial_store<3>(x);
		}

		template<int I>
		LSIMD_ENSURE_INLINE sse_f32pk bsx_pk() const
		{
			return m_pk.bsx<I>();
		}

	public:

		LSIMD_ENSURE_INLINE sse_vec operator + (const sse_vec& rhs) const
		{
			return sse_vec(m_pk + rhs.m_pk);
		}

		LSIMD_ENSURE_INLINE sse_vec operator - (const sse_vec& rhs) const
		{
			return sse_vec(m_pk - rhs.m_pk);
		}

		LSIMD_ENSURE_INLINE sse_vec operator % (const sse_vec& rhs) const
		{
			return sse_vec(m_pk * rhs.m_pk);
		}

		LSIMD_ENSURE_INLINE sse_vec& operator += (const sse_vec& rhs)
		{
			m_pk = m_pk + rhs.m_pk;
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator -= (const sse_vec& rhs)
		{
			m_pk = m_pk - rhs.m_pk;
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator %= (const sse_vec& rhs)
		{
			m_pk = m_pk * rhs.m_pk;
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec operator * (const sse_f32pk& s) const
		{
			return sse_vec(m_pk * s);
		}

		LSIMD_ENSURE_INLINE sse_vec& operator *= (const sse_f32pk& s)
		{
			m_pk = m_pk * s;
			return *this;
		}

	public:
		LSIMD_ENSURE_INLINE f32 sum() const
		{
			return m_pk.partial_sum<3>();
		}

		LSIMD_ENSURE_INLINE f32 dot(const sse_vec& rhs) const
		{
			return (m_pk * rhs.m_pk).partial_sum<3>();
		}


	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return m_pk.test_equal(r[0], r[1], r[2], 0.f);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [3]:\n");
			std::printf("    m_pk = "); m_pk.dump(fmt); std::printf("\n");
		}

	public:
		sse_f32pk m_pk;
	};

	template<> class sse_vec<f32, 4>
	{
	public:
		LSIMD_ENSURE_INLINE explicit sse_vec(const __m128 p) : m_pk(p) { }
		LSIMD_ENSURE_INLINE explicit sse_vec(const sse_f32pk& p) : m_pk(p) { }

	public:
		LSIMD_ENSURE_INLINE sse_vec() { }

		LSIMD_ENSURE_INLINE sse_vec( zero_t ) : m_pk( zero_t() ) { }

		LSIMD_ENSURE_INLINE sse_vec(const f32 e0, const f32 e1, const f32 e2, const f32 e3)
		{
			m_pk.v = _mm_setr_ps(e0, e1, e2, e3);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f32 *x, aligned_t)
		{
			m_pk.load(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE sse_vec(const f32 *x, unaligned_t)
		{
			m_pk.load(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE void load(const f32 *x, aligned_t)
		{
			m_pk.load(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE void load(const f32 *x, unaligned_t)
		{
			m_pk.load(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE void store(f32 *x, aligned_t) const
		{
			m_pk.store(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE void store(f32 *x, unaligned_t) const
		{
			m_pk.store(x, unaligned_t());
		}

		template<int I>
		LSIMD_ENSURE_INLINE sse_f32pk bsx_pk() const
		{
			return m_pk.bsx<I>();
		}

	public:

		LSIMD_ENSURE_INLINE sse_vec operator + (const sse_vec& rhs) const
		{
			return sse_vec(m_pk + rhs.m_pk);
		}

		LSIMD_ENSURE_INLINE sse_vec operator - (const sse_vec& rhs) const
		{
			return sse_vec(m_pk - rhs.m_pk);
		}

		LSIMD_ENSURE_INLINE sse_vec operator % (const sse_vec& rhs) const
		{
			return sse_vec(m_pk * rhs.m_pk);
		}

		LSIMD_ENSURE_INLINE sse_vec& operator += (const sse_vec& rhs)
		{
			m_pk = m_pk + rhs.m_pk;
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator -= (const sse_vec& rhs)
		{
			m_pk = m_pk - rhs.m_pk;
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator %= (const sse_vec& rhs)
		{
			m_pk = m_pk * rhs.m_pk;
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec operator * (const sse_f32pk& s) const
		{
			return sse_vec(m_pk * s);
		}

		LSIMD_ENSURE_INLINE sse_vec& operator *= (const sse_f32pk& s)
		{
			m_pk = m_pk * s;
			return *this;
		}

	public:
		LSIMD_ENSURE_INLINE f32 sum() const
		{
			return m_pk.sum();
		}

		LSIMD_ENSURE_INLINE f32 dot(const sse_vec& rhs) const
		{
			return (m_pk * rhs.m_pk).sum();
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return m_pk.test_equal(r[0], r[1], r[2], r[3]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [4]:\n");
			std::printf("    m_pk = "); m_pk.dump(fmt); std::printf("\n");
		}

	public:
		sse_f32pk m_pk;
	};



	/********************************************
	 *
	 *  sse_vec class for f64
	 *
	 ********************************************/

	template<> class sse_vec<f64, 1>
	{
	public:
		LSIMD_ENSURE_INLINE explicit sse_vec(const __m128d p) : m_pk(p) { }
		LSIMD_ENSURE_INLINE explicit sse_vec(const sse_f64pk& p) : m_pk(p) { }

	public:
		LSIMD_ENSURE_INLINE sse_vec() { }

		LSIMD_ENSURE_INLINE sse_vec( zero_t ) : m_pk( zero_t() ) { }

		LSIMD_ENSURE_INLINE sse_vec(const f64 e0)
		{
			m_pk.v = _mm_set_sd(e0);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f64 *x, aligned_t)
		{
			m_pk.partial_load<1>(x);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f64 *x, unaligned_t)
		{
			m_pk.partial_load<1>(x);
		}

		LSIMD_ENSURE_INLINE void load(const f64 *x, aligned_t)
		{
			m_pk.partial_load<1>(x);
		}

		LSIMD_ENSURE_INLINE void load(const f64 *x, unaligned_t)
		{
			m_pk.partial_load<1>(x);
		}

		LSIMD_ENSURE_INLINE void store(f64 *x, aligned_t) const
		{
			m_pk.partial_store<1>(x);
		}

		LSIMD_ENSURE_INLINE void store(f64 *x, unaligned_t) const
		{
			m_pk.partial_store<1>(x);
		}

		template<int I>
		LSIMD_ENSURE_INLINE sse_f64pk bsx_pk() const
		{
			return m_pk.bsx<I>();
		}

	public:

		LSIMD_ENSURE_INLINE sse_vec operator + (const sse_vec& rhs) const
		{
			return sse_vec(add_s(m_pk, rhs.m_pk));
		}

		LSIMD_ENSURE_INLINE sse_vec operator - (const sse_vec& rhs) const
		{
			return sse_vec(sub_s(m_pk, rhs.m_pk));
		}

		LSIMD_ENSURE_INLINE sse_vec operator % (const sse_vec& rhs) const
		{
			return sse_vec(mul_s(m_pk, rhs.m_pk));
		}

		LSIMD_ENSURE_INLINE sse_vec& operator += (const sse_vec& rhs)
		{
			m_pk = add_s(m_pk, rhs.m_pk);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator -= (const sse_vec& rhs)
		{
			m_pk = sub_s(m_pk, rhs.m_pk);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator %= (const sse_vec& rhs)
		{
			m_pk = mul_s(m_pk, rhs.m_pk);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec operator * (const sse_f64pk& s) const
		{
			return sse_vec(mul_s(m_pk, s));
		}

		LSIMD_ENSURE_INLINE sse_vec& operator *= (const sse_f64pk& s)
		{
			m_pk = mul_s(m_pk, s);
			return *this;
		}

	public:
		LSIMD_ENSURE_INLINE f64 sum() const
		{
			return m_pk.to_scalar();
		}

		LSIMD_ENSURE_INLINE f64 dot(const sse_vec& rhs) const
		{
			return _mm_cvtsd_f64(_mm_mul_sd(m_pk.v, rhs.m_pk.v));
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return m_pk.test_equal(r[0], 0.0);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [1]:\n");
			std::printf("    m_pk = "); m_pk.dump(fmt); std::printf("\n");
		}

	public:
		sse_f64pk m_pk;
	};


	template<> class sse_vec<f64, 2>
	{
	public:
		LSIMD_ENSURE_INLINE explicit sse_vec(const __m128d p) : m_pk(p) { }
		LSIMD_ENSURE_INLINE explicit sse_vec(const sse_f64pk& p) : m_pk(p) { }

	public:
		LSIMD_ENSURE_INLINE sse_vec() { }

		LSIMD_ENSURE_INLINE sse_vec( zero_t ) : m_pk( zero_t() ) { }

		LSIMD_ENSURE_INLINE sse_vec(const f64 e0, const f64 e1)
		{
			m_pk.v = _mm_setr_pd(e0, e1);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f64 *x, aligned_t)
		{
			m_pk.load(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE sse_vec(const f64 *x, unaligned_t)
		{
			m_pk.load(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE void load(const f64 *x, aligned_t)
		{
			m_pk.load(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE void load(const f64 *x, unaligned_t)
		{
			m_pk.load(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE void store(f64 *x, aligned_t) const
		{
			m_pk.store(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE void store(f64 *x, unaligned_t) const
		{
			m_pk.store(x, unaligned_t());
		}

		template<int I>
		LSIMD_ENSURE_INLINE sse_f64pk bsx_pk() const
		{
			return m_pk.bsx<I>();
		}

	public:

		LSIMD_ENSURE_INLINE sse_vec operator + (const sse_vec& rhs) const
		{
			return sse_vec(m_pk + rhs.m_pk);
		}

		LSIMD_ENSURE_INLINE sse_vec operator - (const sse_vec& rhs) const
		{
			return sse_vec(m_pk - rhs.m_pk);
		}

		LSIMD_ENSURE_INLINE sse_vec operator % (const sse_vec& rhs) const
		{
			return sse_vec(m_pk * rhs.m_pk);
		}

		LSIMD_ENSURE_INLINE sse_vec& operator += (const sse_vec& rhs)
		{
			m_pk = m_pk + rhs.m_pk;
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator -= (const sse_vec& rhs)
		{
			m_pk = m_pk - rhs.m_pk;
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator %= (const sse_vec& rhs)
		{
			m_pk = m_pk * rhs.m_pk;
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec operator * (const sse_f64pk& s) const
		{
			return sse_vec(m_pk * s);
		}

		LSIMD_ENSURE_INLINE sse_vec& operator *= (const sse_f64pk& s)
		{
			m_pk = m_pk * s;
			return *this;
		}

	public:
		LSIMD_ENSURE_INLINE f64 sum() const
		{
			return m_pk.sum();
		}

		LSIMD_ENSURE_INLINE f64 dot(const sse_vec& rhs) const
		{
			return (m_pk * rhs.m_pk).sum();
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return m_pk.test_equal(r[0], r[1]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [2]:\n");
			std::printf("    m_pk = "); m_pk.dump(fmt); std::printf("\n");
		}

	public:
		sse_f64pk m_pk;
	};

		
	namespace sse
	{
		template<int I> struct _sse_v2d_bsx;

		template<> struct _sse_v2d_bsx<0>
		{
			LSIMD_ENSURE_INLINE
			static sse_f64pk get(const sse_f64pk& pk0, const sse_f64pk& )
			{
				return pk0.bsx<0>();
			}
		};

		template<> struct _sse_v2d_bsx<1>
		{
			LSIMD_ENSURE_INLINE
			static sse_f64pk get(const sse_f64pk& pk0, const sse_f64pk& )
			{
				return pk0.bsx<1>();
			}
		};

		template<> struct _sse_v2d_bsx<2>
		{
			LSIMD_ENSURE_INLINE
			static sse_f64pk get(const sse_f64pk&, const sse_f64pk& pk1)
			{
				return pk1.bsx<0>();
			}
		};

		template<> struct _sse_v2d_bsx<3>
		{
			LSIMD_ENSURE_INLINE
			static sse_f64pk get(const sse_f64pk&, const sse_f64pk& pk1)
			{
				return pk1.bsx<1>();
			}
		};
	}


	template<> class sse_vec<f64, 3>
	{
	public:
		LSIMD_ENSURE_INLINE explicit sse_vec(const sse_f64pk& pk0, const sse_f64pk& pk1)
		: m_pk0(pk0), m_pk1(pk1) { }

	public:
		LSIMD_ENSURE_INLINE sse_vec() { }

		LSIMD_ENSURE_INLINE sse_vec( zero_t ) : m_pk0( zero_t() ), m_pk1( zero_t() ) { }

		LSIMD_ENSURE_INLINE sse_vec(const f64 e0, const f64 e1, const f64 e2)
		{
			m_pk0.v = _mm_setr_pd(e0, e1);
			m_pk1.v = _mm_set_sd(e2);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f64 *x, aligned_t)
		{
			_load(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE sse_vec(const f64 *x, unaligned_t)
		{
			_load(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE void load(const f64 *x, aligned_t)
		{
			_load(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE void load(const f64 *x, unaligned_t)
		{
			_load(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE void store(f64 *x, aligned_t) const
		{
			_store(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE void store(f64 *x, unaligned_t) const
		{
			_store(x, unaligned_t());
		}

		template<int I>
		LSIMD_ENSURE_INLINE sse_f64pk bsx_pk() const
		{
			return sse::_sse_v2d_bsx<I>::get(m_pk0, m_pk1);
		}

	public:

		LSIMD_ENSURE_INLINE sse_vec operator + (const sse_vec& rhs) const
		{
			return sse_vec((m_pk0 + rhs.m_pk0), add_s(m_pk1, rhs.m_pk1));
		}

		LSIMD_ENSURE_INLINE sse_vec operator - (const sse_vec& rhs) const
		{
			return sse_vec((m_pk0 - rhs.m_pk0), sub_s(m_pk1, rhs.m_pk1));
		}

		LSIMD_ENSURE_INLINE sse_vec operator % (const sse_vec& rhs) const
		{
			return sse_vec((m_pk0 * rhs.m_pk0), mul_s(m_pk1, rhs.m_pk1));
		}

		LSIMD_ENSURE_INLINE sse_vec& operator += (const sse_vec& rhs)
		{
			m_pk0 = m_pk0 + rhs.m_pk0;
			m_pk1 = add_s(m_pk1, rhs.m_pk1);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator -= (const sse_vec& rhs)
		{
			m_pk0 = m_pk0 - rhs.m_pk0;
			m_pk1 = sub_s(m_pk1, rhs.m_pk1);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator %= (const sse_vec& rhs)
		{
			m_pk0 = m_pk0 * rhs.m_pk0;
			m_pk1 = mul_s(m_pk1, rhs.m_pk1);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec operator * (const sse_f64pk& s) const
		{
			return sse_vec(m_pk0 * s, mul_s(m_pk1, s));
		}

		LSIMD_ENSURE_INLINE sse_vec& operator *= (const sse_f64pk& s)
		{
			m_pk0 = m_pk0 * s;
			m_pk1 = mul_s(m_pk1, s);
			return *this;
		}

	public:
		LSIMD_ENSURE_INLINE f64 sum() const
		{
			return (m_pk0 + m_pk1).sum();
		}

		LSIMD_ENSURE_INLINE f64 dot(const sse_vec& rhs) const
		{
			return (operator %(rhs)).sum();
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return m_pk0.test_equal(r[0], r[1]) && m_pk1.test_equal(r[2], 0.0);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [3]:\n");
			std::printf("    m_pk0 = "); m_pk0.dump(fmt); std::printf("\n");
			std::printf("    m_pk1 = "); m_pk1.dump(fmt); std::printf("\n");
		}

	private:
		template<typename AlignT>
		LSIMD_ENSURE_INLINE
		void _load(const f64 *x, AlignT)
		{
			m_pk0.load(x, AlignT());
			m_pk1.partial_load<1>(x + 2);
		}


		template<typename AlignT>
		LSIMD_ENSURE_INLINE
		void _store(f64 *x, AlignT) const
		{
			m_pk0.store(x, AlignT());
			m_pk1.partial_store<1>(x + 2);
		}

	public:
		sse_f64pk m_pk0;
		sse_f64pk m_pk1;
	};


	template<> class sse_vec<f64, 4>
	{
	public:
		LSIMD_ENSURE_INLINE explicit sse_vec(const sse_f64pk& pk0, const sse_f64pk& pk1)
		: m_pk0(pk0), m_pk1(pk1) { }

	public:
		LSIMD_ENSURE_INLINE sse_vec() { }

		LSIMD_ENSURE_INLINE sse_vec( zero_t ) : m_pk0( zero_t() ), m_pk1( zero_t() ) { }

		LSIMD_ENSURE_INLINE sse_vec(const f64 e0, const f64 e1, const f64 e2, const f64 e3)
		{
			m_pk0.v = _mm_setr_pd(e0, e1);
			m_pk1.v = _mm_setr_pd(e2, e3);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f64 *x, aligned_t)
		{
			_load(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE sse_vec(const f64 *x, unaligned_t)
		{
			_load(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE void load(const f64 *x, aligned_t)
		{
			_load(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE void load(const f64 *x, unaligned_t)
		{
			_load(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE void store(f64 *x, aligned_t) const
		{
			_store(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE void store(f64 *x, unaligned_t) const
		{
			_store(x, unaligned_t());
		}

		template<int I>
		LSIMD_ENSURE_INLINE sse_f64pk bsx_pk() const
		{
			return sse::_sse_v2d_bsx<I>::get(m_pk0, m_pk1);
		}

	public:

		LSIMD_ENSURE_INLINE sse_vec operator + (const sse_vec& rhs) const
		{
			return sse_vec((m_pk0 + rhs.m_pk0), (m_pk1 + rhs.m_pk1));
		}

		LSIMD_ENSURE_INLINE sse_vec operator - (const sse_vec& rhs) const
		{
			return sse_vec((m_pk0 - rhs.m_pk0), (m_pk1 - rhs.m_pk1));
		}

		LSIMD_ENSURE_INLINE sse_vec operator % (const sse_vec& rhs) const
		{
			return sse_vec((m_pk0 * rhs.m_pk0), (m_pk1 * rhs.m_pk1));
		}

		LSIMD_ENSURE_INLINE sse_vec& operator += (const sse_vec& rhs)
		{
			m_pk0 = m_pk0 + rhs.m_pk0;
			m_pk1 = m_pk1 + rhs.m_pk1;
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator -= (const sse_vec& rhs)
		{
			m_pk0 = m_pk0 - rhs.m_pk0;
			m_pk1 = m_pk1 - rhs.m_pk1;
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator %= (const sse_vec& rhs)
		{
			m_pk0 = m_pk0 * rhs.m_pk0;
			m_pk1 = m_pk1 * rhs.m_pk1;
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec operator * (const sse_f64pk& s) const
		{
			return sse_vec( m_pk0 * s, m_pk1 * s);
		}

		LSIMD_ENSURE_INLINE sse_vec& operator *= (const sse_f64pk& s)
		{
			m_pk0 = m_pk0 * s;
			m_pk1 = m_pk1 * s;
			return *this;
		}

	public:
		LSIMD_ENSURE_INLINE f64 sum() const
		{
			return (m_pk0 + m_pk1).sum();
		}

		LSIMD_ENSURE_INLINE f64 dot(const sse_vec& rhs) const
		{
			return (operator %(rhs)).sum();
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return m_pk0.test_equal(r[0], r[1]) && m_pk1.test_equal(r[2], r[3]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [4]:\n");
			std::printf("    m_pk0 = "); m_pk0.dump(fmt); std::printf("\n");
			std::printf("    m_pk1 = "); m_pk1.dump(fmt); std::printf("\n");
		}

	private:
		template<typename AlignT>
		LSIMD_ENSURE_INLINE
		void _load(const f64 *x, AlignT)
		{
			m_pk0.load(x, AlignT());
			m_pk1.load(x + 2, AlignT());
		}


		template<typename AlignT>
		LSIMD_ENSURE_INLINE
		void _store(f64 *x, AlignT) const
		{
			m_pk0.store(x, AlignT());
			m_pk1.store(x + 2, AlignT());
		}

	public:
		sse_f64pk m_pk0;
		sse_f64pk m_pk1;
	};

	/** @} */

}

#endif /* SSE_VEC_H_ */
