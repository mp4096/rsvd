# rsvd

[![Build Status](https://travis-ci.org/mp4096/rsvd.svg?branch=master)](https://travis-ci.org/mp4096/rsvd)

Randomized singular value decomposition (SVD) written in C++14 / Eigen.

## Requirements

* gcc ≥ 5.4 or clang ≥ 3.8 (clang 6.0 is recommended)
* CMake ≥ 3.5
* Eigen ≥ 3.3

## Thank you

* Markus Herb [@herb](https://github.com/mherb) for the [`kalman`](https://github.com/mherb/kalman) library which served as an example for good C++ style.
* Kazuya Otani [@kazuotani14](https://github.com/kazuotani14) for the [`RandomizedSvd`](https://github.com/kazuotani14/RandomizedSvd) C++ / Eigen implementation.
* [`scikit-learn`](https://github.com/scikit-learn/scikit-learn) and [`fbpca`](https://github.com/facebook/fbpca) contributors for the reference randomized SVD implementations in Python.
