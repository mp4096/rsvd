#include <limits>

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <rsvd/Constants.hpp>
#include <rsvd/ErrorEstimators.hpp>
#include <rsvd/RandomizedSvd.hpp>

using Eigen::Index;

template <typename T> struct RandomizedSvd : public ::testing::Test {
  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  const Index numRows = 50;
  const Index numCols = 25;
  const Index rank = 5;
  const unsigned int prngSeed = 777;
};

using NumericalTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;

TYPED_TEST_CASE(RandomizedSvd, NumericalTypes, );

TYPED_TEST(RandomizedSvd, ExactRankApproximationNoConditioner) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);
  const MatrixType m = MatrixType::Random(TestFixture::numRows, TestFixture::rank) *
                       MatrixType::Random(TestFixture::rank, TestFixture::numCols);

  Eigen::JacobiSVD<MatrixType> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const MatrixType reconstructedSvd =
      svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().adjoint();
  const auto relErrSvd = Rsvd::relativeFrobeniusNormError(m, reconstructedSvd);

  std::mt19937_64 randomEngine;
  randomEngine.seed(TestFixture::prngSeed);
  Rsvd::RandomizedSvd<MatrixType, std::mt19937_64, Rsvd::SubspaceIterationConditioner::None> rsvd(
      randomEngine);
  rsvd.compute(m, TestFixture::rank);
  const MatrixType reconstructedRsvd =
      rsvd.matrixU() * rsvd.singularValues().asDiagonal() * rsvd.matrixV().adjoint();
  const auto relErrRsvd = Rsvd::relativeFrobeniusNormError(m, reconstructedRsvd);

  ASSERT_LE(relErrRsvd, 1.15 * relErrSvd);
}

TYPED_TEST(RandomizedSvd, ExactRankApproximationLuConditioner) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);
  const MatrixType m = MatrixType::Random(TestFixture::numRows, TestFixture::rank) *
                       MatrixType::Random(TestFixture::rank, TestFixture::numCols);

  Eigen::JacobiSVD<MatrixType> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const MatrixType reconstructedSvd =
      svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().adjoint();
  const auto relErrSvd = Rsvd::relativeFrobeniusNormError(m, reconstructedSvd);

  std::mt19937_64 randomEngine;
  randomEngine.seed(TestFixture::prngSeed);
  Rsvd::RandomizedSvd<MatrixType, std::mt19937_64, Rsvd::SubspaceIterationConditioner::Lu> rsvd(randomEngine);
  rsvd.compute(m, TestFixture::rank);
  const MatrixType reconstructedRsvd =
      rsvd.matrixU() * rsvd.singularValues().asDiagonal() * rsvd.matrixV().adjoint();
  const auto relErrRsvd = Rsvd::relativeFrobeniusNormError(m, reconstructedRsvd);

  ASSERT_LE(relErrRsvd, 1.05 * relErrSvd);
}

TYPED_TEST(RandomizedSvd, ExactRankApproximationMgsConditioner) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);
  const MatrixType m = MatrixType::Random(TestFixture::numRows, TestFixture::rank) *
                       MatrixType::Random(TestFixture::rank, TestFixture::numCols);

  Eigen::JacobiSVD<MatrixType> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const MatrixType reconstructedSvd =
      svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().adjoint();
  const auto relErrSvd = Rsvd::relativeFrobeniusNormError(m, reconstructedSvd);

  std::mt19937_64 randomEngine;
  randomEngine.seed(TestFixture::prngSeed);
  Rsvd::RandomizedSvd<MatrixType, std::mt19937_64, Rsvd::SubspaceIterationConditioner::Mgs> rsvd(randomEngine);
  rsvd.compute(m, TestFixture::rank);
  const MatrixType reconstructedRsvd =
      rsvd.matrixU() * rsvd.singularValues().asDiagonal() * rsvd.matrixV().adjoint();
  const auto relErrRsvd = Rsvd::relativeFrobeniusNormError(m, reconstructedRsvd);

  ASSERT_LE(relErrRsvd, 1.30 * relErrSvd);
}

TYPED_TEST(RandomizedSvd, OversamplingMgsConditioner) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);
  const MatrixType m = MatrixType::Random(TestFixture::numRows, TestFixture::rank) *
                       MatrixType::Random(TestFixture::rank, TestFixture::numCols);

  Eigen::JacobiSVD<MatrixType> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const MatrixType reconstructedSvd =
      svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().adjoint();
  const auto relErrSvd = Rsvd::relativeFrobeniusNormError(m, reconstructedSvd);

  std::mt19937_64 randomEngine;
  randomEngine.seed(TestFixture::prngSeed);
  Rsvd::RandomizedSvd<MatrixType, std::mt19937_64, Rsvd::SubspaceIterationConditioner::Mgs> rsvd(randomEngine);
  // Oversample (twice the rank)
  rsvd.compute(m, TestFixture::rank, TestFixture::rank);
  const MatrixType reconstructedRsvd =
      rsvd.matrixU() * rsvd.singularValues().asDiagonal() * rsvd.matrixV().adjoint();
  const auto relErrRsvd = Rsvd::relativeFrobeniusNormError(m, reconstructedRsvd);

  ASSERT_LE(relErrRsvd, 1.30 * relErrSvd);
}

TYPED_TEST(RandomizedSvd, ExactRankApproximationQrConditioner) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);
  const MatrixType m = MatrixType::Random(TestFixture::numRows, TestFixture::rank) *
                       MatrixType::Random(TestFixture::rank, TestFixture::numCols);

  Eigen::JacobiSVD<MatrixType> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const MatrixType reconstructedSvd =
      svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().adjoint();
  const auto relErrSvd = Rsvd::relativeFrobeniusNormError(m, reconstructedSvd);

  std::mt19937_64 randomEngine;
  randomEngine.seed(TestFixture::prngSeed);
  Rsvd::RandomizedSvd<MatrixType, std::mt19937_64, Rsvd::SubspaceIterationConditioner::Qr> rsvd(
      randomEngine);
  rsvd.compute(m, TestFixture::rank);
  const MatrixType reconstructedRsvd =
      rsvd.matrixU() * rsvd.singularValues().asDiagonal() * rsvd.matrixV().adjoint();
  const auto relErrRsvd = Rsvd::relativeFrobeniusNormError(m, reconstructedRsvd);

  ASSERT_LE(relErrRsvd, 1.05 * relErrSvd);
}
