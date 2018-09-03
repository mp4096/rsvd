#include <complex>
#include <limits>

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <rsvd/StandardNormalRandom.hpp>

using Eigen::Index;
using Rsvd::Internal::standardNormalRandom;

template <typename T> struct StandardNormalRandom : public ::testing::Test {
  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using RealType = typename Eigen::NumTraits<T>::Real;

  const Index numRows = 10;
  const Index numCols = 1;
  const Index numTrials = 1'000;
  const unsigned int prngSeed = 777;
  const RealType tol = 1e-1;
};

using NumericalTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;

TYPED_TEST_CASE(StandardNormalRandom, NumericalTypes, );

TYPED_TEST(StandardNormalRandom, ZeroMean) {
  using MatrixType = typename TestFixture::MatrixType;

  MatrixType acc = MatrixType::Zero(TestFixture::numRows, TestFixture::numCols);

  std::mt19937_64 randomEngine;
  randomEngine.seed(TestFixture::prngSeed);
  for (unsigned int i = 0; i < TestFixture::numTrials; ++i) {
    acc += standardNormalRandom<MatrixType, std::mt19937_64>(TestFixture::numRows,
                                                             TestFixture::numCols, randomEngine);
  }
  acc /= TestFixture::numTrials;

  // Since the vector has 10 independent elements, we want to check for the worst case among them
  ASSERT_LE(acc.cwiseAbs().maxCoeff(), TestFixture::tol);
}

TYPED_TEST(StandardNormalRandom, Covariance) {
  using MatrixType = typename TestFixture::MatrixType;

  MatrixType x = MatrixType::Zero(TestFixture::numRows, TestFixture::numTrials);

  std::mt19937_64 randomEngine;
  randomEngine.seed(TestFixture::prngSeed);
  for (unsigned int i = 0; i < TestFixture::numTrials; ++i) {
    x.col(i) = standardNormalRandom<MatrixType, std::mt19937_64>(
        TestFixture::numRows, TestFixture::numCols, randomEngine);
  }
  MatrixType covariance = x * x.adjoint() / TestFixture::numTrials;

  const typename MatrixType::Scalar unitVariance = 1;
  // Check diagonal entries: They should be almost equal to one (unit variance)
  for (Index i = 0; i < TestFixture::numRows; ++i) {
    ASSERT_LE(std::abs(covariance(i, i) - unitVariance), TestFixture::tol);
  }

  // Remove diagonal entries
  covariance.diagonal().setZero();
  // Check all remaining entries: They should be almost zero (independent distributions)
  ASSERT_LE(covariance.cwiseAbs().maxCoeff(), TestFixture::tol);
}
