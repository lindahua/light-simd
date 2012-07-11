/**
 * @file simd_mat.h
 *
 * @brief SIMD-based fixed-size matrix classes
 *
 * @author Dahua Lin
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


	template<typename T, int M, int N, typename Kind>
	struct simd_mat_traits;

	template<typename T, int M, int N>
	struct simd_mat_traits<T, M, N, sse_kind>
	{
		typedef sse_mat<T, M, N> impl_type;
	};

	/**
	 * @brief Generic fixed size matrix.
	 */
	template<typename T, int M, int N, typename Kind>
	struct simd_mat
	{
		typedef T value_type;
		typedef typename simd_mat_traits<T, M, N, Kind>::impl_type impl_type;
		typedef simd_pack<T, Kind> pack_type;
		
		impl_type impl;

		// constructors

		LSIMD_ENSURE_INLINE
		simd_mat() { }

		LSIMD_ENSURE_INLINE
		simd_mat( zero_t ) : impl( zero_t() ) { }

		LSIMD_ENSURE_INLINE
		simd_mat( const impl_type& imp ) : impl(imp) { }

		LSIMD_ENSURE_INLINE
		simd_mat(const T *x, aligned_t)
		: impl(x, aligned_t()) { }

		LSIMD_ENSURE_INLINE
		simd_mat(const T *x, unaligned_t)
		: impl(x, unaligned_t()) { }

		LSIMD_ENSURE_INLINE
		simd_mat(const T *x, int ldim, aligned_t)
		: impl(x, ldim, aligned_t()) { }

		LSIMD_ENSURE_INLINE
		simd_mat(const T *x, int ldim, unaligned_t)
		: impl(x, ldim, unaligned_t()) { }


		// load and store

		LSIMD_ENSURE_INLINE
		void load(const T *x, aligned_t)
		{
			impl.load(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load(const T *x, unaligned_t)
		{
			impl.load(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load(const T *x, int ldim, aligned_t)
		{
			impl.load(x, ldim, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load(const T *x, int ldim, unaligned_t)
		{
			impl.load(x, ldim, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load_trans(const T *x, aligned_t)
		{
			impl.load_trans(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load_trans(const T *x, unaligned_t)
		{
			impl.load_trans(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load_trans(const T *x, int ldim, aligned_t)
		{
			impl.load_trans(x, ldim, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load_trans(const T *x, int ldim, unaligned_t)
		{
			impl.load_trans(x, ldim, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void store(T *x, aligned_t) const
		{
			impl.store(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void store(T *x, unaligned_t) const
		{
			impl.store(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void store(T *x, int ldim, aligned_t) const
		{
			impl.store(x, ldim, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void store(T *x, int ldim, unaligned_t) const
		{
			impl.store(x, ldim, unaligned_t());
		}

		// linear operations

		LSIMD_ENSURE_INLINE
		simd_mat operator + (const simd_mat& r) const
		{
			return impl + r.impl;
		}

		LSIMD_ENSURE_INLINE
		simd_mat operator - (const simd_mat& r) const
		{
			return impl - r.impl;
		}

		LSIMD_ENSURE_INLINE
		simd_mat operator % (const simd_mat& r) const
		{
			return impl % r.impl;
		}

		LSIMD_ENSURE_INLINE
		simd_mat operator * (const pack_type& s) const
		{
			return impl * s.impl;
		}

		LSIMD_ENSURE_INLINE
		simd_mat& operator += (const simd_mat& r)
		{
			impl += r.impl;
			return *this;
		}

		LSIMD_ENSURE_INLINE
		simd_mat& operator -= (const simd_mat& r)
		{
			impl -= r.impl;
			return *this;
		}

		LSIMD_ENSURE_INLINE
		simd_mat& operator %= (const simd_mat& r)
		{
			impl %= r.impl;
			return *this;
		}

		LSIMD_ENSURE_INLINE
		simd_mat& operator *= (const pack_type& s)
		{
			impl *= s.impl;
			return *this;
		}


		LSIMD_ENSURE_INLINE
		simd_vec<T, M> operator * (const simd_vec<T, N>& v) const
		{
			return impl * v.impl;
		}

		LSIMD_ENSURE_INLINE
		T trace() const
		{
			return impl.trace();
		}

	};


	template<typename Kind, typename T, int M, int K, int N>
	inline simd_mat<T, M, N, Kind> operator * (
			const simd_mat<T, M, K, Kind>& A,
			const simd_mat<T, K, N, Kind>& B)
	{
		simd_mat<T, M, N, Kind> C;
		C.impl = A.impl * B.impl;
		return C;
	}

	template<typename Kind, typename T, int N>
	inline T det(const simd_mat<T, N, N, Kind>& A)
	{
		return det(A.impl);
	}

	template<typename Kind, typename T, int N>
	inline simd_mat<T, N, N, Kind> inv(const simd_mat<T, N, N, Kind>& A)
	{
		return inv(A.impl);
	}

	template<typename Kind, typename T, int N>
	inline T inv_and_det(const simd_mat<T, N, N, Kind>& A, simd_mat<T, N, N, Kind>& R)
	{
		return inv_and_det(A.impl, R.impl);
	}

	template<typename Kind, typename T, int N>
	inline simd_vec<T, N, Kind> solve(const simd_mat<T, N, N, Kind>& A, simd_vec<T, N, Kind> b)
	{
		return solve(A.impl, b.impl);
	}

	template<typename Kind, typename T, int N, int N2>
	inline simd_mat<T, N, N2, Kind> solve(const simd_mat<T, N, N, Kind>& A, const simd_mat<T, N, N2, Kind>& B)
	{
		return solve(A.impl, B.impl);
	}

}

#endif /* SIMD_MAT_H_ */
