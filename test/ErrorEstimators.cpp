#include <complex>
#include <limits>

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <rsvd/ErrorEstimators.hpp>

using Rsvd::relativeFrobeniusNormError;

template <typename T> struct RelativeFrobeniusNormError : public ::testing::Test {
  using RealType = typename Eigen::NumTraits<T>::Real;
  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  const RealType macheps = std::numeric_limits<RealType>::epsilon();
};

using NumericalTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;

TYPED_TEST_CASE(RelativeFrobeniusNormError, NumericalTypes, );

/// \brief Relative error between equal matrices must be zero.
TYPED_TEST(RelativeFrobeniusNormError, SameMatrix) {
  using MatrixType = typename TestFixture::MatrixType;

  const MatrixType a = MatrixType::Identity(4, 4);
  ASSERT_NEAR(relativeFrobeniusNormError(a, a), 0, TestFixture::macheps);
}

TYPED_TEST(RelativeFrobeniusNormError, DifferentMatrices) {
  using MatrixType = typename TestFixture::MatrixType;

  MatrixType reference = MatrixType::Zero(3, 4);
  reference(0, 0) = 1;

  MatrixType approx = MatrixType::Zero(3, 4);
  approx(0, 0) = 1;
  approx(2, 3) = 100;

  const auto relErr = relativeFrobeniusNormError(reference, approx);
  ASSERT_NEAR(relErr, 100, TestFixture::macheps);
}

TYPED_TEST(RelativeFrobeniusNormError, NonCommutativity) {
  using MatrixType = typename TestFixture::MatrixType;

  MatrixType a = MatrixType::Zero(2, 2);
  a << 1, 2, 3, 4;

  MatrixType b = MatrixType::Zero(2, 2);
  b << 5, 6, 7, 8;

  const auto relativeToA = relativeFrobeniusNormError(a, b);
  const auto relativeToB = relativeFrobeniusNormError(b, a);

  ASSERT_NEAR(relativeToA, 8 / sqrt(30), TestFixture::macheps);
  ASSERT_NEAR(relativeToB, 8 / sqrt(174), TestFixture::macheps);
}

/// \brief Relative error function asserts that the reference matrix has non-zero norm.
TYPED_TEST(RelativeFrobeniusNormError, ZeroReferenceMatrix) {
  using MatrixType = typename TestFixture::MatrixType;

  const MatrixType a = MatrixType::Zero(4, 4);
  ASSERT_DEATH(relativeFrobeniusNormError(a, a), "Assertion `referenceNorm > 0' failed");
}
