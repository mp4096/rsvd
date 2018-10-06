# rsvd

[![Build Status](https://travis-ci.org/mp4096/rsvd.svg?branch=master)](https://travis-ci.org/mp4096/rsvd)

Randomized singular value decomposition (SVD) written in C++14 / Eigen.

## What is a randomized SVD?

Let *A* be a real or complex matrix of size _m ⨉ n_, _rank(A) = r_.
Its economic singular value decomposition (SVD) is given by
_A = UΣV*_ with the following factor matrices:

* left singular vectors matrix *U* of size _m ⨉ r_
* singular values matrix *Σ* of size _r ⨉ r_
* right singular vectors matrix *V* of size _n ⨉ r_

The left and right singular vector matrices have orthonormal columns:
_U*U = I_, _V*V = I_.
The singular values matrix *Σ* is diagonal and
has sorted singular values on its principal diagonal.

When *m* and *n* are very large, we want to approximate this SVD using the randomized
range approximation *Q* of size _m ⨉ r_ such that _|| A - QQ*A ||₂ < ε_.
Then, the problem can be projected onto a smaller subspace as follows: _B = Q*A_,
*B* is of size _r ⨉ n_.
After the SVD of the smaller problem _B = ŨΣV*_,
the solution to the original problem can be recovered as _U = QŨ_.

The range *Q* is approximated using random sampling.
In order to capture the largest singular values, randomized subspace iterations can be used.
However, as any power iteration methods, they suffer from numerical problems.
To mitigate this problem, the user can select an appropriate conditioner based on the
modified Gram–Schmidt process, the LU decomposition, and the QR decomposition.
The conditioner choice is a trade-off between runtime and numerical properties.

## Why should I use `mp4096/rsvd`?

Well, first of all make sure you _need_ to use C++.
RSVD implementations in Python
([`scikit-learn/scikit-learn`](https://github.com/scikit-learn/scikit-learn),
[`facebook/fbpca`](https://github.com/facebook/fbpca)) are
competitive performance-wise and much easier to use.

Still want to use C++?

Then you might want to use `mp4096/rsvd` because:

* :rocket: it is compile-time generic over real and complex, single and double precision matrices
* :mortar_board: it supports one-shot range approximation as well as randomized subspace iterations
  with a choice of conditioners (none, MGS, LU, QR)
* :book: it has nice Doxygen documentation
* :microscope: it is well-tested
* :game_die: it offers you fine control over random number generation for the sampling matrix `Ω`,
  thus alleviating any potential problems with non-deterministic approximation results
  due to different PRNG seeds
* :+1: it is written in idiomatic C++14 style

## Requirements

* gcc ≥ 5.4 or clang ≥ 3.8 (clang 6.0 is recommended)
* CMake ≥ 3.5
* Eigen ≥ 3.3

## Installation

This is a header-only library, just make sure you have Eigen installed and
add the following includes to your program:

```cpp
#include <rsvd/Constants.hpp>
#include <rsvd/ErrorEstimators.hpp>
#include <rsvd/RandomizedSvd.hpp>
```

Unfortunately, a header-only library requires longer compile times,
since it is recompiled every time.

## Usage

See `examples/SimpleUsage/main.cpp`. You can compile it as follows:

```
$ mkdir -p build
$ cd build
$ cmake ..
$ make -j2 example_simple_usage
```

## References

* [Nathan Halko, Per-Gunnar Martinsson, Joel A. Tropp. _Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. 2010._](https://arxiv.org/abs/0909.4061)


## Thank you

* Markus Herb [@herb](https://github.com/mherb) for the [`kalman`](https://github.com/mherb/kalman) library which served as an example for good C++ style.
* Kazuya Otani [@kazuotani14](https://github.com/kazuotani14) for the [`RandomizedSvd`](https://github.com/kazuotani14/RandomizedSvd) C++ / Eigen implementation.
* [`scikit-learn`](https://github.com/scikit-learn/scikit-learn) and [`fbpca`](https://github.com/facebook/fbpca) contributors for the reference randomized SVD implementations in Python.
