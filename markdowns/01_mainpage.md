# Light SIMD 

_A light-weight C++ library for SIMD-based computation_

## What is SIMD

**SIMD (Single Instruction, Multiple Data)** generally refers to instruction sets to support data-parallel computation. An SIMD instruction may operate on multiple data simultaneously within one or a few CPU cycles. 
Many numerical algorithms involve repeatedly applying the same sequence of math operations to thousands or millions of data values. In this programs, proper use of SIMD can often lead to substantial improvement of run-time performance. 

Most modern computers support SIMD instruction sets. 
[SSE (Streaming SIMD Extensions)](http://en.wikipedia.org/wiki/Streaming_SIMD_Extensions), which was introduced in 1999 by Intel in their Pentium III processors, is probably one of the most widely used SIMD instruction architecture. SSE was subsequently expanded to SSE2, SSE3, SSSE3, and SSE4. 

SSE has eight or sixteen 128-bit XMM registers, which may hold four single-precision or two double-precision floating-point numbers. An SSE instruction can perform operations on two XMM registers (_e.g._ adding four single-precision real numbers with another four at the same time).  In 2012, Intel introduced [AVX (Advanced Vector Extensions)](http://en.wikipedia.org/wiki/Advanced_Vector_Extensions) with the Sandy bridge processors. AVX has sixteen 256-bit YMM registers, which can accommodate up to eight single-precision or four double precision floating point numbers. Though introduced by Intel, SSE and AVX are also supported by AMD processors.

## Overview of Light SIMD

Light SIMD is a C++ template library that provides flexible and portable API for writing SIMD codes. In addition, it has the following features.

* Generic interfaces for writing platform-independent codes.

* Support of a rich set of operations, which include
  * arithmetic calculations
  * power functions: ``sqrt``(x^(1/2)), ``sqr``(x^(2)), ``rcp``(x^(-1)), ``rsqrt``(x^(-1/2)), ``cube``(x^(3)), ``cbrt``(x^(1/3)), and ``pow``
  * rounding functions: ``floor``, ``ceil``, and ``round``
  * exponential and logarithm functions: ``exp``, ``exp2``, ``exp10``, ``log``, ``log2``, and ``log10``
  * trigonometric functions: ``sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``, ``atan2``
  * hyperbolic functions: ``sinh``, ``cosh``, ``tanh``, ``asinh``, ``acosh``, and ``atanh``
  * other math functions: ``hypot``, ``erf``, and ``erfc``
  * comparison and logical operations 
  * reduction functions: ``sum``, ``max``, and ``min``
  * entry manipulation, _e.g._ swizzling, shuffling, shifting, and broadcasting, etc
  
* Linear algebra module, which contains SIMD-based classes to represent small vectors and matrices. This module also implements a variety of linear algebraic computation with hand optimized codes
  * template classes to represent small fixed-size vectors and matrices
  * specialized codes optimized for vectors of length ``2``, ``3``, or ``4``.
  * specialized codes optimized for matrices of size ``2 x 2``, ``2 x 3``, ``2 x 4``, ``3 x 2``, ``3 x 3``, ``3 x 4``, ``4 x 2``, ``4 x 3``, and ``4 x 4``
  * matrix-vector and matrix-matrix products
  * evaluation of trace and determinant
  * matrix inversion
  * linear equation solving
  
* Light weight classes and always-inlined functions, which incur zero run-time overhead (in release mode). 

* Extensively tested using hundreds of unit testing cases.
  

## Write SIMD Codes

Whereas modern CPUs provide SIMD instruction set architecture, your program, if written in traditional way, would not _magically_ use SIMD. To leverage the efficiency of SIMD instructions, you have to explicitly write SIMD codes. If you are using C/C++, there are three ways to do this:

### Inline assembly

One can take advantage of the inline assembly syntax in C/C++ to directly invoke SIMD instructions. 
Here is a good [tutorial](http://www.3dbuzz.com/vbforum/showthread.php?104753-HowTo-Inline-Assembly-amp-SSE-Vector-normalization-done-fast!) on how to write SSE codes via inline assembly. 
Generally, this requires very low-level stuff such as hand-coded register allocation and stack management, and it is nontrivial to write portable ASM codes.

### Intrinsics

An **intrinsic** is a built-in function that the compiler would directly maps to one or more assembly instructions. Intel specifies a large collection of intrinsics for SSE and AVX, which many modern compilers provide (nearly) complete support. As an example to demonstrate the use of SSE intrinsics, the following is a simple function that calculates ``y += a * x`` on arrays with n entries (here, we simply assume n is a multiple of 4 for simplicity).

    void vadd_prod_f32(unsigned n, float a, const float *x, float *y)
    {
        __m128 pa = _mm_set1_ps(a);
    
        unsigned m = n / 4;
        for (unsigned i = 0; i < m; ++i, x += 4, y += 4)
        {
            __m128 px = _mm_loadu_ps(x);
            __m128 py = _mm_loadu_ps(y);
        
            py = _mm_add_ps(py, _mm_mul_ps(pa, px));
            _mm_storeu_ps(y, py);
        } 
    }


A modified function can be used for double-precision numbers:

```C++
void vadd_prod_f64(unsigned n, double a, const double *x, double *y)
{
    __m128d pa = _mm_set1_pd(a);
    
    unsigned m = n / 2;
    for (unsigned i = 0; i < m; ++i, x += 2, y += 2)
    {
        __m128d px = _mm_loadu_pd(x);
        __m128d py = _mm_loadu_pd(y);
        
        py = _mm_add_pd(py, _mm_mul_pd(pa, px));
        _mm_storeu_pd(y, py);
    } 
}
```

Compared to inline assembly, the codes using intrinsics are easier to read and write. However, this way is not very satisfactory. It is limited in several aspects:
* The SSE data types and function names for floats and doubles are different. The same task need to be implemented for different data types respectively.
* The intrinsic function names such as ``_mm_loadu_ps`` and ``_mm_set1_pd`` are not very intuitive.
* The functions above are **not portable** to other architectures, such as AVX. Basically, one needs to re-implement the functions under different architectures.

### High-level generic library

Through C++ template specialization mechanism, Light-SIMD provides generic interfaces for SIMD programming, where the architecture-dependent details are encapsulated. With Light-SIMD, users can write portable SIMD codes much more easily and cleanly. Below shows how a generic function to calculate ``y += a * x`` can be implemented using Light-SIMD.

```C++
using namespace lsimd;

template<typename T>
void vadd_prod(unsigned n, T a, const T *x, T *y)
{
    const unsigned wid = simd<T>::pack_width;
    const unsigned m = n / wid;
    
    simd_pack<T> pa(x, unaligned_t());
    for (unsigned i = 0; i < m; ++i, x += wid, y += wid)
    {
        simd_pack<T> px(x, unaligned_t());
        simd_pack<T> py(y, unaligned_t());
        
        py += pa * px;
        py.store(y, unaligned_t());
    }
}
```

The code here is cleaner. More importantly, it is portable and can be used for different data types and on different instruction set architectures.
For example, 

```C++
float af, *xf, *yf; 
double ad, *xd, *yd;
// *** some code to initialize the variables
vadd_prod(n, af, xf, yf);
vadd_prod(n, ad, xd, yd);
```

The library will resolve the proper instruction set architecture to use, depending on compilation settings. This frees the developer of the tedious job of writing low-level architecture-dependent codes. 


