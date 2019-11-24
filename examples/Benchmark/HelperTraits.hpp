#ifndef RSVD_BENCHMARK_HELPER_TRAITS_HPP_
#define RSVD_BENCHMARK_HELPER_TRAITS_HPP_

#include <Eigen/Dense>
#include <rsvd/Constants.hpp>

namespace Benchmark {

template <typename T> constexpr auto kMatrixTypeName{""};
template <> constexpr auto kMatrixTypeName<Eigen::MatrixXf>{"float"};
template <> constexpr auto kMatrixTypeName<Eigen::MatrixXd>{"double"};
template <> constexpr auto kMatrixTypeName<Eigen::MatrixXcf>{"complex float"};
template <> constexpr auto kMatrixTypeName<Eigen::MatrixXcd>{"complex double"};

template <Rsvd::SubspaceIterationConditioner Conditioner> constexpr auto kConditionerName{""};
template <>
constexpr auto kConditionerName<Rsvd::SubspaceIterationConditioner::None>{"no conditioner"};
template <>
constexpr auto kConditionerName<Rsvd::SubspaceIterationConditioner::Mgs>{"MGS conditioner"};
template <>
constexpr auto kConditionerName<Rsvd::SubspaceIterationConditioner::Lu>{"LU conditioner"};
template <>
constexpr auto kConditionerName<Rsvd::SubspaceIterationConditioner::Qr>{"QR conditioner"};

} // namespace Benchmark

#endif
