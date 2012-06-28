/**
 * @file sse_pack.h
 *
 * @brief The SSE pack classes and a set of convenient routines.
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

#ifndef LSIMD_SSE_PACK_H_
#define LSIMD_SSE_PACK_H_

#include "details/sse_pack_bits.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4141)
#endif

namespace lsimd
{

	/**
	 * @defgroup packs_sse SSE Packs
	 * @ingroup packs
	 *
	 * @brief SSE-based pack classes
	 */ 	
	/** @{ */ 

	template<typename T> struct sse_pack;

	/**
	 * @brief SSE pack with four single-precision real values.
	 */
	template<>
	struct sse_pack<f32>
	{

		/**
		 * The scalar value type.
		 */
		typedef f32 value_type;

		/**
		 * The builtin representation type.
		 */
		typedef __m128 intern_type;

		/**
		 * The number of scalars in a pack.
		 */
		static const unsigned int pack_width = 4;

		union
		{
			__m128 v;  /**< The builtin representation.  */
			LSIMD_ALIGN_SSE f32 e[4];  /**< The representation in an array of scalars. */
		};


		// constructors

		/**
		 * Default constructor.
		 *
		 * The entries in this pack are left uninitialized.
		 */
		LSIMD_ENSURE_INLINE sse_pack() { }

		/**
		 * Constructs a pack using builtin representation.
		 *
		 * @param v_   The builtin representation of a pack.
		 */
		LSIMD_ENSURE_INLINE sse_pack(const __m128 v_)
		: v(v_) { }

		/**
		 * Constructs a pack with all entries initialized to zeros.
		 *
		 * @post  This pack == (0, 0, 0, 0).
		 */
		LSIMD_ENSURE_INLINE sse_pack( zero_t )
		{
			v = _mm_setzero_ps();
		}

		/**
		 * Constructs a pack with all entries initialized
		 * to a given value.
		 *
		 * @param x   The value used to initialize the pack.
		 *
		 * @post  This pack == (x, x, x, x).
		 */
		LSIMD_ENSURE_INLINE explicit sse_pack(const f32 x)
		{
			v = _mm_set1_ps(x);
		}

		/**
		 * Constructs a pack with given values.
		 *
		 * @param e0   The value to be set to this->e[0].
		 * @param e1   The value to be set to this->e[1]. 
		 * @param e2   The value to be set to this->e[2].
		 * @param e3   The value to be set to this->e[3].
		 *
		 * @post  This pack == (e0, e1, e2, e3).
		 */
		LSIMD_ENSURE_INLINE sse_pack(const f32 e0, const f32 e1, const f32 e2, const f32 e3)
		{
			v = _mm_set_ps(e3, e2, e1, e0);
		}

		/**
		 * Constructs a pack by loading the entry values from
		 * a properly aligned memory address.
		 *
		 * @param a   The memory address from which values are loaded.
		 *
		 * @post  This pack == (a[0], a[1], a[2], a[3]).
		 */
		LSIMD_ENSURE_INLINE sse_pack(const f32* a, aligned_t)
		{
			v = _mm_load_ps(a);
		}

		/**
		 * Constructs a pack by loading the entry values from
		 * a memory address that is not necessarily aligned.
		 *
		 * @param a   The memory address from which values are loaded.
		 *
		 * @post  This pack == (a[0], a[1], a[2], a[3]).
		 */
		LSIMD_ENSURE_INLINE sse_pack(const f32* a, unaligned_t)
		{
			v = _mm_loadu_ps(a);
		}


		/**
		 * @name Basic Information Retrieval Methods
		 *
		 * The member functions to get basic information about the SIMD pack.
		 */
		///@{


		/**
		 * Get the pack width (the number of scalars in a pack).
		 *
		 * @return   The value of \ref pack_width, which equals 4 here.
		 */
		LSIMD_ENSURE_INLINE unsigned int width() const
		{
			return pack_width;
		}

		/**
		 * Get the builtin representation.
		 *
		 * @return   A copy of the builtin representation variable.
		 */
		LSIMD_ENSURE_INLINE __m128 intern() const
		{
			return v;
		}

		///@}


		/**
		 * @name Import and Export Methods
		 *
		 * The member functions to set, load and store entry values.
		 */
		///@{

		/**
		 * Set all scalar entries to zeros.
		 *
		 * @post This pack == (0, 0, 0, 0).
		 */
		LSIMD_ENSURE_INLINE void set_zero()
		{
			v = _mm_setzero_ps();
		}

		/**
		 * Set all scalar entries to a given value.
		 *
		 * @param x the value to be set to all entries.
		 */
		LSIMD_ENSURE_INLINE void set(const f32 x)
		{
			v = _mm_set1_ps(x);
		}

		/**
		 * Set given values to the entries.
		 *
		 * @param e0   The value to be set to this->e[0].
		 * @param e1   The value to be set to this->e[1].
		 * @param e2   The value to be set to this->e[2].
		 * @param e3   The value to be set to this->e[3].
		 *
		 * @post  This pack == (e0, e1, e2, e3).
		 */
		LSIMD_ENSURE_INLINE void set(const f32 e0, const f32 e1, const f32 e2, const f32 e3)
		{
			v = _mm_set_ps(e3, e2, e1, e0);
		}

		/**
		 * Load all entries from an aligned memory address.
		 *
		 * @param a  The memory address from which the values are loaded.
		 *
		 * @post   This pack == (a[0], a[1], a[2], a[3]).
		 */
		LSIMD_ENSURE_INLINE void load(const f32* a, aligned_t)
		{
			v = _mm_load_ps(a);
		}

		/**
		 * Load all entries from an memory address that is not
		 * necessarily aligned.
		 *
		 * @param a  The memory address from which the values are loaded.
		 *
		 * @post     This pack == (a[0], a[1], a[2], a[3]).
		 */
		LSIMD_ENSURE_INLINE void load(const f32* a, unaligned_t)
		{
			v = _mm_loadu_ps(a);
		}

		/**
		 * Store all entries to a properly aligned memory address.
		 *
		 * @param a   The memory address from which the values are stored.
		 *
		 * @post  a[0:3] == this->e[0:3].
		 */
		LSIMD_ENSURE_INLINE void store(f32* a, aligned_t) const
		{
			_mm_store_ps(a, v);
		}

		/**
		 * Store all entries to the memory address that is not
		 * necessarily aligned.
		 *
		 * @param a   The memory address from which the values are stored.
		 *
		 * @post  a[0:3] == this->e[0:3].
		 */
		LSIMD_ENSURE_INLINE void store(f32* a, unaligned_t) const
		{
			_mm_storeu_ps(a, v);
		}

		/**
		 * Load a subset of entries from a given memory address.
		 *
		 * @tparam I   The number of entries to be loaded.
		 *             The value of I must be either 1, 2, or 3.
		 *
		 * @param a    The memory address from which the values
		 *             are loaded.
		 *
		 * @remark   The loaded values are set to the lower-end of
		 *           the pack, while the entries at higher-end are
		 *           set to zeros.
		 *
		 * @post  this->e[0:I-1] == a[0:I-1] &&
		 *        this->e[I:3] == all zeros.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE void partial_load(const f32 *a)
		{
			v = sse::partial_load<I>(a);
		}

		/**
		 * Store a subset of entries to a given memory address.
		 *
		 * @tparam I   The number of entries to be stored.
		 *             The value of I must be either 1, 2, or 3.
		 *
		 * @param a    The memory address to which the values are stored.
		 *
		 * @remark     This method stores the first I values of the pack.
		 *
		 * @post  a[0:I-1] == this->e[0:I-1].
		 */
		template<int I>
		LSIMD_ENSURE_INLINE void partial_store(f32 *a) const
		{
			sse::partial_store<I>(a, v);
		}

		///@}



		/**
		 * @name Entry Manipulation Methods
		 *
		 * The member functions to extract entries or switch their positions.
		 */
		///@{

		/**
		 * Extract the first entry value.
		 *
		 * @return  The value of the first entry (i.e. \ref e[0]).
		 *
		 * @remark  To extract the scalar at arbitrary position,
		 *          one may use another member function \ref extract.
		 *
		 * @see extract.
		 */
		LSIMD_ENSURE_INLINE f32 to_scalar() const
		{
			return _mm_cvtss_f32(v);
		}

		/**
		 * Extract the entry at given position.
		 *
		 * @tparam I the entry position.
		 *           The value of I must be within [0, 3].
		 *
		 * @return the I-th entry of this pack.
		 *
		 * @remark extract<0>() is equivalent to to_scalar().
		 *
		 * @see to_scalar.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE f32 extract() const
		{
			return sse::f32p_extract<I>(v);
		}

		/**
		 * Broadcast the entry at a given position.
		 *
		 * @tparam I  The position of the entry to be broadcasted.
		 *            The value of I must be within [0, 3].
		 *
		 * @return    The resultant pack whose entries are all 
		 *            equal to the I-th entry of this pack,  
		 *            i.e. (e[I], e[I], e[I], e[I]).
		 */
		template<int I>
		LSIMD_ENSURE_INLINE sse_pack bsx() const
		{
			return _mm_shuffle_ps(v, v, _MM_SHUFFLE(I, I, I, I));
		}

		/**
		 * Shift entries towards the low end
		 * (with zeros shift-in from the high end).
		 *
		 * @tparam I  The distance to shift (in terms of the number
		 *            of scalars).
		 *            The value of I must be within [0, 4].
		 *
		 * @return    The shifted pack, of which the k-th
		 *            entry equals the (k+I)-th entry of this pack,
		 *            when k < 4 - I, or zero otherwise.
		 *
		 * @see shift_back.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE sse_pack shift_front() const
		{
			return _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v), (I << 2)));
		}

		/**
		 * Shift entries towards the high end
		 * (with zeros shift-in from the low end).
		 *
		 * @tparam I  The distance to shift (in terms of the number of scalars).
		 *            The value of I must be within [0, 4].
		 *
		 * @return  The shifted pack, of which the k-th
		 *          entry equals the (k-I)-th entry of this pack,
		 *          when k >= I, or zero otherwise.
		 *
		 * @see shift_front.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE sse_pack shift_back() const
		{
			return _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(v), (I << 2)));
		}


		/**
		 * Switch the entry positions.
		 *
		 * @tparam I0  The source position of the entry to be with index 0.
		 * @tparam I1  The source position of the entry to be with index 1.
		 * @tparam I2  The source position of the entry to be with index 2.
		 * @tparam I3  The source position of the entry to be with index 3.
		 *
		 * @return     The resultant pack, as (e[I0], e[I1], e[I2], e[I3]).
		 *
		 * @remark The values of I0, I1, I2, and I3 must be within [0, 3].
		 */
		template<int I0, int I1, int I2, int I3>
		LSIMD_ENSURE_INLINE sse_pack swizzle() const
		{
			return _mm_shuffle_ps(v, v, _MM_SHUFFLE(I3, I2, I1, I0));
		}

		/**
		 * Duplicate the lower half of the pack.
		 *
		 * @return  The resultant pack, as (e[0], e[0], e[1], e[1]).
		 *
		 * @remark  This is functionally equivalent to
		 *          \ref swizzle< 0, 1, 0, 1 >, but can be
		 *          more efficient on some architecture.
		 *
		 * @see swizzle, dup_high, dup2_low, dup2_high.
		 */
		LSIMD_ENSURE_INLINE sse_pack dup_low() const 
		{
			return sse::f32_dup_low(v);
		}

		/**
		 * Duplicate the higher half of the pack.
		 *
		 * @return  The resultant pack, as (e[2], e[2], e[3], e[3]).
		 *
		 * @remark  This is functionally equivalent to
		 *          \ref swizzle< 2, 3, 2, 3 >, but can be
		 *          more efficient on some architecture.
		 *
		 * @see swizzle, dup_low, dup2_low, dup2_high.
		 */
		LSIMD_ENSURE_INLINE sse_pack dup_high() const 
		{
			return sse::f32_dup_high(v);
		}

		/**
		 * Duplicate the lower entry of each half of the pack.
		 *
		 * @return  The resultant pack, as (e[0], e[0], e[2], e[2]).
		 *
		 * @remark  This is functionally equivalent to
		 *          \ref swizzle< 0, 0, 2, 2 >, but can be
		 *          more efficient on some architecture.
		 *
		 * @see swizzle, dup_low, dup_high, dup2_high.
		 */
		LSIMD_ENSURE_INLINE sse_pack dup2_low() const 
		{
			return sse::f32_dup2_low(v);
		}

		/**
		 * Duplicate the higher entry of each half of the pack.
		 *
		 * @return  The resultant pack, as (e[1], e[1], e[3], a[3]).
		 *
		 * @remark  This is functionally equivalent to
		 *          \ref swizzle< 1, 1, 3, 3 >, but can be
		 *          more efficient on some architecture.
		 *
		 * @see swizzle, dup_low, dup_high, dup2_low.
		 */
		LSIMD_ENSURE_INLINE sse_pack dup2_high() const
		{
			return sse::f32_dup2_high(v);
		}


		///@}


		/**
		 * @name Statistics Methods
		 *
		 * The member functions to evaluate statistics over entries.
		 */
		///@{


		/**
		 * Evaluate the sum of all entries.
		 *
		 * @return  The sum of all entries.
		 */
		LSIMD_ENSURE_INLINE f32 sum() const
		{
			return sse::f32_sum(v);
		}

		/**
		 * Evaluate the sum of a subset of entries.
		 *
		 * @tparam I   The number of entries to be used.
		 *             The value of I must be within [1, 3].
		 *
		 * @return     The sum of first I entries from the lowest end.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE f32 partial_sum() const
		{
			return sse::f32_partial_sum<I>(v);
		}

		/**
		 * Evaluate the maximum of a subset of entries.
		 *
		 * @return   The maximum of first I entries from the lowest end.
		 */
		LSIMD_ENSURE_INLINE f32 (max)() const
		{
			return sse::f32_max(v);
		}

		/**
		 * Evaluate the maximum of a subset of entries.
		 *
		 * @tparam I   The number of entries to be used.
		 *             The value of I must be within [1, 3].
		 *
		 * @return     The maximum of first I entries from the lowest end.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE f32 partial_max() const
		{
			return sse::f32_partial_max<I>(v);
		}

		/**
		 * Evaluate the minimum of a subset of entries.
		 *
		 * @return   The minimum of all entries.
		 */
		LSIMD_ENSURE_INLINE f32 (min)() const
		{
			return sse::f32_min(v);
		}

		/**
		 * Evaluate the minimum of a subset of entries.
		 *
		 * @tparam I   The number of entries to be used.
		 *
		 * @return     The minimum of the first I entries from the lowest end.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE f32 partial_min() const
		{
			return sse::f32_partial_min<I>(v);
		}

		///@}


		/**
		 * Constant Generating Methods
		 *
		 * The static member functions to generate packs comprised
		 * of some common useful values.
		 */
		///@{


		/**
		 * Get an all-zero pack.
		 *
		 * @returns   A pack with all entries being zeros, 
		 *            as (0, 0, 0, 0).
		 */
		LSIMD_ENSURE_INLINE static sse_pack zeros()
		{
			return _mm_setzero_ps();
		}

		/**
		 * Get an all-one pack.
		 *
		 * @returns   A pack with all entries being ones,
		 *            as (1, 1, 1, 1).
		 */
		LSIMD_ENSURE_INLINE static sse_pack ones()
		{
			return _mm_set1_ps(1.f);
		}

		/**
		 * Get an all-two pack.
		 *
		 * @returns   A pack with all entries being twos,
		 *            as (2, 2, 2, 2).
		 */
		LSIMD_ENSURE_INLINE static sse_pack twos()
		{
			return _mm_set1_ps(2.f);
		}

		/**
		 * Get an all-half pack.
		 *
		 * @returns   A pack with all entries being 0.5,
		 *            as (0.5, 0.5, 0.5, 0.5).
		 */
		LSIMD_ENSURE_INLINE static sse_pack halfs()
		{
			return _mm_set1_ps(0.5f);
		}

		///@}


		// Only for debug

		LSIMD_ENSURE_INLINE bool test_equal(f32 e0, f32 e1, f32 e2, f32 e3) const
		{
			return e[0] == e0 && e[1] == e1 && e[2] == e2 && e[3] == e3;
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return test_equal(r[0], r[1], r[2], r[3]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("(");
			std::printf(fmt, e[0]); std::printf(", ");
			std::printf(fmt, e[1]); std::printf(", ");
			std::printf(fmt, e[2]); std::printf(", ");
			std::printf(fmt, e[3]);
			std::printf(")");
		}

	}; // end struct sse_pack<f32>


	/**
	 * @brief SSE pack with two double-precision real values.
	 */
	template<>
	struct sse_pack<f64>
	{

		/**
		 * The scalar value type.
		 */
		typedef f64 value_type;

		/**
		 * The builtin representation type.
		 */
		typedef __m128d intern_type;

		/**
		 * The number of scalar in each pack.
		 */
		static const unsigned int pack_width = 2;

		union
		{
			__m128d v; /**< The builtin representation. */
			LSIMD_ALIGN_SSE f64 e[2];  /** The representation in an array of scalars. */
		};

		
		/**
		 * Default constructor.
		 *
		 * The entries in the pack are left uninitialized.
		 */
		LSIMD_ENSURE_INLINE sse_pack() { }

		/**
		 * Constructs a pack using builtin representation.
		 *
		 * @param v_  The builtin representation of a pack.
		 */
		LSIMD_ENSURE_INLINE sse_pack(const intern_type v_)
		: v(v_) { }

		/**
		 * Constructs a pack initialized as all zeros.
		 *
		 * @post  This pack == (0, 0, 0, 0).
		 */		
		LSIMD_ENSURE_INLINE sse_pack( zero_t )
		{
			v = _mm_setzero_pd();
		}

		/**
		 * Constructs a pack with all scalar entries initialized
		 * to a given value.
		 *
		 * @param x  The value used to initialize the pack.
		 *
		 * @post This pack == (x, x, x, x).
		 */
		LSIMD_ENSURE_INLINE explicit sse_pack(const f64 x)
		{
			v = _mm_set1_pd(x);
		}

		/**
		 * Constructs a pack with given values.
		 *
		 * @param e0   The value to be set to this->e[0].
		 * @param e1   The value to be set to this->e[1].
		 *
		 * @post This pack == (e0, e1).
		 */
		LSIMD_ENSURE_INLINE sse_pack(const f64 e0, const f64 e1)
		{
			v = _mm_set_pd(e1, e0);
		}

		/**
		 * Constructs a pack by loading the entry values from
		 * a properly aligned memory address.
		 *
		 * @param a  The memory address from which values are loaded.
		 *
		 * @post  This pack == (a[0], a[1]).
		 */
		LSIMD_ENSURE_INLINE sse_pack(const f64* a, aligned_t)
		{
			v = _mm_load_pd(a);
		}

		/**
		 * Constructs a pack by loading the entry values from
		 * a memory address that is not necessarily aligned.
		 *
		 * @param a The memory address from which values are loaded.
		 *
		 * @post  This pack == (a[0], a[1]).
		 */
		LSIMD_ENSURE_INLINE sse_pack(const f64* a, unaligned_t)
		{
			v = _mm_loadu_pd(a);
		}


		/**
		 * @name Basic Information Retrieval Methods.
		 *
		 * The member functions to get basic information about the SIMD pack.
		 */
		///@{

		/**
		 * Get the pack width (the number of scalars in a pack).
		 *
		 * @return the value of \ref pack_width.
		 */
		LSIMD_ENSURE_INLINE unsigned int width() const
		{
			return pack_width;
		}

		/**
		 * Get the builtin representation.
		 *
		 * @return   A copy of the builtin representation variable.
		 */
		LSIMD_ENSURE_INLINE intern_type intern() const
		{
			return v;
		}

		///@}


		/**
		 * @name Import and Export Methods
		 *
		 * The member functions to set, load and store entry values.
		 */
		///@{

		/**
		 * Set all scalar entries to zeros.
		 *
		 * @post  This pack == (0, 0). 
		 */
		LSIMD_ENSURE_INLINE void set_zero()
		{
			v = _mm_setzero_pd();
		}

		/**
		 * Set all scalar entries to a given value.
		 *
		 * @param x  The value to be set to all entries.
		 *
		 * @post  This pack == (x, x). 
		 */
		LSIMD_ENSURE_INLINE void set(const f64 x)
		{
			v = _mm_set1_pd(x);
		}

		/**
		 * Set entries to given values.
		 *
		 * @param e0  The value to be set to this->e[0].
		 * @param e1  The value to be set to this->e[1].
		 *
		 * @post  This pack == (e0, e1).
		 */
		LSIMD_ENSURE_INLINE void set(const f64 e0, const f64 e1)
		{
			v = _mm_set_pd(e1, e0);
		}

		/**
		 * Load all entries from an aligned memory address.
		 *
		 * @param a  The memory address from which the values are loaded.
		 *
		 * @post  This pack == (a[0], a[1]).
		 */
		LSIMD_ENSURE_INLINE void load(const f64* a, aligned_t)
		{
			v = _mm_load_pd(a);
		}

		/**
		 * Load all entries from an memory address that is not
		 * necessarily aligned.
		 *
		 * @param a  The memory address from which the values are loaded.
		 *
		 * @post  This pack == (a[0], a[1]).
		 */
		LSIMD_ENSURE_INLINE void load(const f64* a, unaligned_t)
		{
			v = _mm_loadu_pd(a);
		}

		/**
		 * Store all entries to a properly aligned memory address.
		 *
		 * @param a  The memory address to which the values are stored.
		 */
		LSIMD_ENSURE_INLINE void store(f64* a, aligned_t) const
		{
			_mm_store_pd(a, v);
		}

		/**
		 * Store all entries to the memory address that is not
		 * necessarily aligned.
		 *
		 * @param a  The memory address from which the values
		 *           are stored.
		 */
		LSIMD_ENSURE_INLINE void store(f64* a, unaligned_t) const
		{
			_mm_storeu_pd(a, v);
		}

		/**
		 * Load a subset of entries from a given memory address
		 *
		 * @tparam I  The number of entries to be loaded.
		 *            The value of I must be 1 here.
		 *
		 * @param a   The memory address from which the value is loaded.
		 *
		 * @post  This pack == (a[0], 0).
		 *
		 * @remark  The address a need not be aligned here.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE void partial_load(const f64 *a)
		{
			v = sse::partial_load<I>(a);
		}

		/**
		 * Store a subset of entries to a given memory address
		 *
		 * @tparam I  The number of entries to be stored.
		 *            The value of I must be 1 here.
		 *
		 * @param a   The memory address to which the value is stored.
		 *
		 * @post  a[0] == this->e[0].
		 *
		 * @remark  The address a need not be aligned here.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE void partial_store(f64 *a) const
		{
			sse::partial_store<I>(a, v);
		}

		///@}

		/**
		 * @name Entry Manipulation Methods
		 *
		 * The member functions to extract entries or switch their positions.
		 */
		///@{

		/**
		 * Extract the first entry (the one with index zero).
		 *
		 * @return  The scalar value of the first entry, i.e. e[0].
		 *
		 * @remark To extract the scalar at arbitrary position,
		 *         one may use another member function \ref extract.
		 *
		 * @see  extract.
		 */
		LSIMD_ENSURE_INLINE f64 to_scalar() const
		{
			return _mm_cvtsd_f64(v);
		}

		/**
		 * Extract the entry at a given position.
		 *
		 * @tparam I  The entry position.
		 *            The value of I must be either 0 or 1.
		 *
		 * @return the I-th entry of this pack.
		 *
		 * @remark extract<0>() is equivalent to to_scalar().
		 *
		 * @see to_scalar.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE f64 extract() const
		{
			return sse::f64p_extract<I>(v);
		}

		/**
		 * Broadcast the entry at a given position.
		 *
		 * @tparam I  The position of the entry to be broadcasted.
		 *            The value of I must be either 0 or 1.
		 *
		 * @return    A pack whose entries are all equal to
		 *            the I-th entry of this pack.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE sse_pack bsx() const
		{
			return _mm_shuffle_pd(v, v, _MM_SHUFFLE2(I, I));
		}

		/**
		 * Shift entries towards the low end
		 * (with zeros shift-in from the high end)
		 *
		 * @tparam I  The distance to shift (in terms of the number
		 *            of scalars).
		 *            The value of I must be within [0, 2].
		 *
		 * @return  The resultant pack, of which the k-th
		 *          entry equals the (k+I)-th entry of this pack,
		 *          when k < 2 - I, or zero otherwise.
		 *
		 * @see shift_front.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE sse_pack shift_front() const
		{
			return _mm_castsi128_pd(_mm_srli_si128(_mm_castpd_si128(v), (I << 3)));
		}

		/**
		 * Shift entries towards the high end
		 * (with zeros shift-in from the low end).
		 *
		 * @tparam I   The distance to shift (in terms of the number
		 *             of scalars).
		 *             The value of I must be within [0, 2].
		 *
		 * @return   The shifted pack, of which the k-th
		 *           entry equals the (k-I)-th entry of this pack,
		 *           when k >= I, or zero otherwise.
		 *
		 * @see shift_front.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE sse_pack shift_back() const
		{
			return _mm_castsi128_pd(_mm_slli_si128(_mm_castpd_si128(v), (I << 3)));
		}

		/**
		 * Switch the entry positions.
		 *
		 * @tparam I0  The source position of the entry to be with index 0.
		 * @tparam I1  The source position of the entry to be with index 1.
		 *
		 * @return     The resultant pack, as (e[I0], e[I1]).
		 *
		 * @remark The values of I0, I1, I2, and I3 must be either 0 or 1.
		 */
		template<int I0, int I1>
		LSIMD_ENSURE_INLINE sse_pack swizzle() const
		{
			return _mm_shuffle_pd(v, v, _MM_SHUFFLE2(I1, I0));
		}

		/**
		 * Duplicate the lower entry of the pack.
		 *
		 * @return  The resultant pack, as (e[0], e[0]).
		 *
		 * @remark  This is functionally equivalent to
		 *          \ref swizzle< 0, 0 >, but can be
		 *          more efficient on some architecture.
		 *
		 * @see swizzle, dup_high.
		 */
		LSIMD_ENSURE_INLINE sse_pack dup_low() const 
		{
			return sse::f64_dup_low(v);
		}

		/**
		 * Duplicate the higher entry of the pack.
		 *
		 * @return  The resultant pack, as (e[1], e[1]).
		 *
		 * @remark  This is functionally equivalent to
		 *          \ref swizzle< 1, 1 >, but can be
		 *          more efficient on some architecture.
		 *
		 * @see swizzle, dup_high.
		 */
		LSIMD_ENSURE_INLINE sse_pack dup_high() const 
		{
			return sse::f64_dup_high(v);
		}

		///@}


		// statistics

		/**
		 * @name Statistics Methods
		 *
		 * The member functions to evaluate statistics over entries.
		 */
		///@{

		/**
		 * Evaluate the sum of all entries.
		 *
		 * @return  The sum of all entries (i.e. e[0] + e[1]).
		 */ 
		LSIMD_ENSURE_INLINE f64 sum() const
		{
			return sse::f64_sum(v);
		}

		/**
		 * Evaluate the sum of a subset of entries.
		 *
		 * @tparam I   The number of entries to be used, which
		 *             must equal 1 here. 
		 *
		 * @return     The value of e[0].
		 *
		 * @remark  This function is included for interface consistency.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE f64 partial_sum() const
		{
			return sse::f64_partial_sum<I>(v);
		}

		/**
		 * Evaluate the maximum of all entries.
		 *
		 * @return  The maximum of all entries (i.e. max(e[0], e[1])).
		 */ 
		LSIMD_ENSURE_INLINE f64 (max)() const
		{
			return sse::f64_max(v);
		}

		/**
		 * Evaluate the maximum of a subset of entries.
		 *
		 * @tparam I   The number of entries to be used, which
		 *             must equal 1 here. 
		 *
		 * @return     The value of e[0].
		 *
		 * @remark  This function is included for interface consistency.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE f64 partial_max() const
		{
			return sse::f64_partial_max<I>(v);
		}

		/**
		 * Evaluate the minimum of all entries.
		 *
		 * @return  The minimum of all entries (i.e. min(e[0], e[1])).
		 */ 
		LSIMD_ENSURE_INLINE f64 (min)() const
		{
			return sse::f64_min(v);
		}

		/**
		 * Evaluate the minimum of a subset of entries.
		 *
		 * @tparam I   The number of entries to be used, which
		 *             must equal 1 here. 
		 *
		 * @return     The value of e[0].
		 *
		 * @remark  This function is included for interface consistency.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE f64 partial_min() const
		{
			return sse::f64_partial_min<I>(v);
		}

		///@}


		/**
		 * Constant Generating Methods
		 *
		 * The static member functions to generate packs comprised
		 * of some common useful values.
		 */
		///@{

		/**
		 * Get an all-zero pack.
		 *
		 * @returns   A pack with all entries being zeros, i.e. (0, 0).
		 */
		LSIMD_ENSURE_INLINE static sse_pack zeros()
		{
			return _mm_setzero_pd();
		}

		/**
		 * Get an all-one pack.
		 *
		 * @returns   A pack with all entries being ones, i.e. (1, 1).
		 */
		LSIMD_ENSURE_INLINE static sse_pack ones()
		{
			return _mm_set1_pd(1.0);
		}

		/**
		 * Get an all-two pack.
		 *
		 * @returns   A pack with all entries being twos, i.e. (2, 2).
		 */
		LSIMD_ENSURE_INLINE static sse_pack twos()
		{
			return _mm_set1_pd(2.0);
		}

		/**
		 * Get an all-half pack.
		 *
		 * @returns   A pack with all entries being 0.5, i.e. (0.5, 0.5).
		 */
		LSIMD_ENSURE_INLINE static sse_pack halfs()
		{
			return _mm_set1_pd(0.5);
		}

		///@}


		// Only for debug

		LSIMD_ENSURE_INLINE bool test_equal(f64 e0, f64 e1) const
		{
			return e[0] == e0 && e[1] == e1;
		}

		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return test_equal(r[0], r[1]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("(");
			std::printf(fmt, e[0]); std::printf(", ");
			std::printf(fmt, e[1]);
			std::printf(")");
		}

	}; // end struct sse_pack<f64>


	// typedefs

	/**
	 * @brief A short name for sse_pack<f32>.
	 */
	typedef sse_pack<f32> sse_f32pk;

	/**
	 * @brief A short name for sse_pack<f64>.
	 */
	typedef sse_pack<f64> sse_f64pk;


	// Some auxiliary routines


	/**
	 * Constructing an SSE pack by moving two entries from a to 
	 * the lower part and two from b to the higher part.
     *
     * @tparam I0  The index of the entry in a to be with index 0
     * @tparam I1  The index of the entry in a to be with index 1
     * @tparam I2  The index of the entry in b to be with index 2
     * @tparam I3  The index of the entry in b to be with index 2
     *
     * @param a    The source pack for the lower part.
     * @param b    The source pack for the higher part.
     *
     * @return   The resultant pack as (a[I0], a[I1], b[I2], b[I3]).
	 */
	template<int I0, int I1, int I2, int I3>
	LSIMD_ENSURE_INLINE
	inline sse_f32pk shuffle(const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_shuffle_ps(a.v, b.v, _MM_SHUFFLE(I3, I2, I1, I0));
	}

	/**
	 * Constructing an SSE pack by moving an entry from a to 
	 * the lower part and another from b to the higher part.
     *
     * @tparam I0  The index of the entry in a to be with index 0
     * @tparam I1  The index of the entry in b to be with index 1
     *
     * @param a    The source pack for the lower part.
     * @param b    The source pack for the higher part.
     *
     * @return   The resultant pack as (a[I0], b[I1]).
	 */
	template<int I0, int I1>
	LSIMD_ENSURE_INLINE
	inline sse_f64pk shuffle(const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_shuffle_pd(a.v, b.v, _MM_SHUFFLE2(I1, I0));
	}

	/**
	 * Merge the lower parts of two packs.
	 *
	 * @param a  The first source pack
	 * @param b  The second source pack
	 *
	 * @return   The resultant pack as (a[0], a[1], b[0], b[1]).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk merge_low(const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_movelh_ps(a.v, b.v);
	}

	/**
	 * Merge the higher parts of two packs.
	 *
	 * @param a  The first source pack
	 * @param b  The second source pack
	 *
	 * @return   The resultant pack as (a[2], a[3], b[2], b[3]).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk merge_high(const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_movehl_ps(b.v, a.v);
	}

	/**
	 * Unpack (i.e. interleave) the lower part of two packs
	 *
	 * @param a  The first source pack
	 * @param b  The second source pack
	 *
	 * @return   The resultant pack as (a[0],b[0],a[1],b[1]).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk unpack_low(const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_unpacklo_ps(a.v, b.v);
	}

	/**
	 * Unpack (i.e. interleave) the higher part of two packs
	 *
	 * @param a  The first source pack
	 * @param b  The second source pack
	 *
	 * @return   The resultant pack as (a[2],b[2],a[3],b[3]).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f32pk unpack_high(const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_unpackhi_ps(a.v, b.v);
	}

	/**
	 * Unpack (i.e. interleave) the lower entry of two packs
	 *
	 * @param a  The first source pack
	 * @param b  The second source pack
	 *
	 * @return   The resultant pack as (a[0],b[0]).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk unpack_low(const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_unpacklo_pd(a.v, b.v);
	}

	/**
	 * Unpack (i.e. interleave) the higher entry of two packs
	 *
	 * @param a  The first source pack
	 * @param b  The second source pack
	 *
	 * @return   The resultant pack as (a[1],b[1]).
	 */
	LSIMD_ENSURE_INLINE
	inline sse_f64pk unpack_high(const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_unpackhi_pd(a.v, b.v);
	}

    /** @} */ // packs_sse
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif /* SSE_PACK_H_ */
