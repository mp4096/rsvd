#ifndef RSVD_BENCHMARK_HELPER_TRAITS_HPP_
#define RSVD_BENCHMARK_HELPER_TRAITS_HPP_

#include <Eigen/Dense>

using Eigen::MatrixXcd;
using Eigen::MatrixXcf;
using Eigen::MatrixXd;
using Eigen::MatrixXf;

namespace Benchmark {

template <typename T> struct MatrixTypeName { static const char *get(); };

template <> struct MatrixTypeName<MatrixXf> {
  static const char *get() { return "float"; }
};

template <> struct MatrixTypeName<MatrixXd> {
  static const char *get() { return "double"; }
};

template <> struct MatrixTypeName<MatrixXcf> {
  static const char *get() { return "complex float"; }
};

template <> struct MatrixTypeName<MatrixXcd> {
  static const char *get() { return "complex double"; }
};

} // namespace Benchmark

#endif
