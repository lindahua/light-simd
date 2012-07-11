/**
 * @file simd_pack.h
 *
 * @brief The generic template class for representing SIMD packs.
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SIMD_PACK_H_
#define LSIMD_SIMD_PACK_H_

#include <light_simd/common/common_base.h>
#include <light_simd/sse/sse_pack.h>

namespace lsimd
{
	/**
	 * @brief SIMD type traits for SSE Kind.
	 */
	template<typename T>
	struct simd<T, sse_kind>
	{
		typedef sse_pack<T> impl_type;
		typedef typename impl_type::intern_type intern_type;

		static const unsigned int pack_width = impl_type::pack_width;
		static const unsigned int max_registers = 8;
	};

	template<typename T, typename Kind>
	struct simd_pack
	{
		typedef T value_type;
		typedef typename simd<T, Kind>::impl_type impl_type;
		typedef typename simd<T, Kind>::intern_type intern_type;

		static const unsigned int pack_width = simd<T, Kind>::pack_width;

		impl_type impl;

		// constructors

		LSIMD_ENSURE_INLINE simd_pack() { }
		LSIMD_ENSURE_INLINE simd_pack(const impl_type& imp)
		: impl(imp) { }


		LSIMD_ENSURE_INLINE simd_pack(intern_type v)
		: impl(v) { }

		LSIMD_ENSURE_INLINE simd_pack( zero_t )
		: impl(zero_t()) { }

		LSIMD_ENSURE_INLINE explicit simd_pack(const T x)
		: impl(x) { }

		LSIMD_ENSURE_INLINE simd_pack(const T* a, aligned_t)
		: impl(a, aligned_t()) { }

		LSIMD_ENSURE_INLINE simd_pack(const T* a, unaligned_t)
		: impl(a, unaligned_t()) { }


		// basic getters

		LSIMD_ENSURE_INLINE unsigned int width() const
		{
			return pack_width;
		}

		LSIMD_ENSURE_INLINE intern_type intern() const
		{
			return impl.intern();
		}

		// set, load, and store

		LSIMD_ENSURE_INLINE void set_zero()
		{
			impl.set_zero();
		}

		LSIMD_ENSURE_INLINE void set(const T x)
		{
			impl.set(x);
		}

		LSIMD_ENSURE_INLINE void load(const T* a, aligned_t)
		{
			impl.load(a, aligned_t());
		}

		LSIMD_ENSURE_INLINE void load(const T* a, unaligned_t)
		{
			impl.load(a, unaligned_t());
		}

		LSIMD_ENSURE_INLINE void store(T* a, aligned_t) const
		{
			impl.store(a, aligned_t());
		}

		LSIMD_ENSURE_INLINE void store(T* a, unaligned_t) const
		{
			impl.store(a, unaligned_t());
		}

		template<int I>
		LSIMD_ENSURE_INLINE void partial_load(const T *a)
		{
			impl.partial_load<I>(a);
		}

		template<int I>
		LSIMD_ENSURE_INLINE void partial_store(T *a) const
		{
			impl.partial_store<I>(a);
		}


		// entry manipulation

		LSIMD_ENSURE_INLINE T to_scalar() const
		{
			return impl.to_scalar();
		}

		template<int I>
		LSIMD_ENSURE_INLINE T extract() const
		{
			return impl.extract<I>();
		}

		template<int I>
		LSIMD_ENSURE_INLINE simd_pack bsx() const
		{
			return impl.bsx<I>();
		}

		template<int I>
		LSIMD_ENSURE_INLINE simd_pack shift_front() const
		{
			return impl.shift_front<I>();
		}

		template<int I>
		LSIMD_ENSURE_INLINE simd_pack shift_back() const
		{
			return impl.shift_back<I>();
		}

		// statistics

		LSIMD_ENSURE_INLINE T sum() const
		{
			return impl.sum();
		}

		template<int I>
		LSIMD_ENSURE_INLINE T partial_sum() const
		{
			return impl.partial_sum<I>();
		}

		LSIMD_ENSURE_INLINE T (max)() const
		{
			return (impl.max)();
		}

		template<int I>
		LSIMD_ENSURE_INLINE T partial_max() const
		{
			return impl.partial_max<I>();
		}

		LSIMD_ENSURE_INLINE T (min)() const
		{
			return (impl.min)();
		}

		template<int I>
		LSIMD_ENSURE_INLINE T partial_min() const
		{
			return impl.partial_min<I>();
		}

		// constants

		LSIMD_ENSURE_INLINE static simd_pack false_mask()
		{
			return impl_type::false_mask();
		}

		LSIMD_ENSURE_INLINE static simd_pack true_mask()
		{
			return impl_type::true_mask();
		}

		LSIMD_ENSURE_INLINE static simd_pack zeros()
		{
			return impl_type::zeros();
		}

		LSIMD_ENSURE_INLINE static simd_pack ones()
		{
			return impl_type::ones();
		}

	};

}

#endif /* SIMD_BASE_H_ */









