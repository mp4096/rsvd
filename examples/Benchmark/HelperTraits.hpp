#ifndef RSVD_BENCHMARK_HELPER_TRAITS_HPP_
#define RSVD_BENCHMARK_HELPER_TRAITS_HPP_

#include <Eigen/Dense>
#include <rsvd/Constants.hpp>

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

template <Rsvd::SubspaceIterationConditioner Conditioner>
inline constexpr auto getConditionerName() noexcept;
template <>
inline constexpr auto getConditionerName<Rsvd::SubspaceIterationConditioner::None>() noexcept {
  return "no conditioner";
}
template <>
inline constexpr auto getConditionerName<Rsvd::SubspaceIterationConditioner::Mgs>() noexcept {
  return "MGS conditioner";
}
template <>
inline constexpr auto getConditionerName<Rsvd::SubspaceIterationConditioner::Lu>() noexcept {
  return "LU conditioner";
}
template <>
inline constexpr auto getConditionerName<Rsvd::SubspaceIterationConditioner::Qr>() noexcept {
  return "QR conditioner";
}

} // namespace Benchmark

#endif
