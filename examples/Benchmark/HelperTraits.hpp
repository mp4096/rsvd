#ifndef RSVD_BENCHMARK_HELPER_TRAITS_HPP_
#define RSVD_BENCHMARK_HELPER_TRAITS_HPP_

#include <Eigen/Dense>

namespace Benchmark {

template <typename T> inline constexpr auto getMatrixTypeName() noexcept;
template <> inline constexpr auto getMatrixTypeName<Eigen::MatrixXf>() noexcept { return "float"; }
template <> inline constexpr auto getMatrixTypeName<Eigen::MatrixXd>() noexcept {
  return "double";
}
template <> inline constexpr auto getMatrixTypeName<Eigen::MatrixXcf>() noexcept {
  return "complex float";
}
template <> inline constexpr auto getMatrixTypeName<Eigen::MatrixXcd>() noexcept {
  return "complex double";
}

} // namespace Benchmark

#endif
