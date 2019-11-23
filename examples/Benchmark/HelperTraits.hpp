#ifndef RSVD_BENCHMARK_HELPER_TRAITS_HPP_
#define RSVD_BENCHMARK_HELPER_TRAITS_HPP_

#include <Eigen/Dense>

namespace Benchmark {

template <typename T> struct MatrixTypeName { static const char *get(); };

template <> struct MatrixTypeName<Eigen::MatrixXf> {
  static const char *get() { return "float"; }
};

template <> struct MatrixTypeName<Eigen::MatrixXd> {
  static const char *get() { return "double"; }
};

template <> struct MatrixTypeName<Eigen::MatrixXcf> {
  static const char *get() { return "complex float"; }
};

template <> struct MatrixTypeName<Eigen::MatrixXcd> {
  static const char *get() { return "complex double"; }
};

} // namespace Benchmark

#endif
