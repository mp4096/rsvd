#include <limits>
#include <random>

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <rsvd/Constants.hpp>
#include <rsvd/RandomizedRangeFinder.hpp>

using Eigen::Index;
using Rsvd::LuConditioner;
using Rsvd::MgsConditioner;
using Rsvd::NoConditioner;
using Rsvd::QrConditioner;
using Rsvd::Internal::RandomizedSubspaceIterations;
using Rsvd::Internal::singleShot;

// Note: Tolerances for the range approximation (tests "{SingleShot, NoConditioner, LuConditioner,
// MgsConditioner, QrConditioner}Approximation") were chosen after several runs with different PRNG
// seeds (444, 555, 666, 777).

template <typename T> struct RandomizedRangeFinder : public ::testing::Test {
  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using RealType = typename Eigen::NumTraits<T>::Real;

  const Index numRows = 50;
  const Index numCols = 25;
  const Index dim = 15;
  // The oversampling additional to the rank is important to test for correct deflation, especially
  // for the MGS conditioner
  const Index oversampling = 5;
  const unsigned int numIter = 2;
  const unsigned int prngSeed = 777;

  const RealType macheps = std::numeric_limits<RealType>::epsilon();
};

using NumericalTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;

TYPED_TEST_CASE(RandomizedRangeFinder, NumericalTypes, );

// \brief Range approximation must have columns of unit length
TYPED_TEST(RandomizedRangeFinder, SingleShotNorm) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);
  std::mt19937_64 randomEngine;
  randomEngine.seed(TestFixture::prngSeed);

  // Create a matrix with rank loss
  const MatrixType a = MatrixType::Random(TestFixture::numRows, TestFixture::dim) *
                       MatrixType::Random(TestFixture::dim, TestFixture::numCols);
  // Compute the randomized range approximation
  const MatrixType q = singleShot<MatrixType, std::mt19937_64>(
      a, TestFixture::dim + TestFixture::oversampling, randomEngine);

  for (Index i = 0; i < TestFixture::dim; ++i) {
    ASSERT_NEAR(q.col(i).norm(), 1, 2 * TestFixture::macheps);
  }
}

// \brief Range approximation must have orthogonal columns
TYPED_TEST(RandomizedRangeFinder, SingleShotOrthogonality) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);
  std::mt19937_64 randomEngine;
  randomEngine.seed(TestFixture::prngSeed);

  // Create a matrix with rank loss
  const MatrixType a = MatrixType::Random(TestFixture::numRows, TestFixture::dim) *
                       MatrixType::Random(TestFixture::dim, TestFixture::numCols);
  // Compute the randomized range approximation
  const MatrixType q = singleShot<MatrixType, std::mt19937_64>(
      a, TestFixture::dim + TestFixture::oversampling, randomEngine);

  for (Index i = 0; i < TestFixture::dim; ++i) {
    for (Index j = 0; j < i; ++j) {
      const auto res = std::abs(q.col(i).dot(q.col(j)));
      ASSERT_NEAR(res, 0, 2 * TestFixture::macheps);
    }
  }
}

// \brief Range approximation must satisfy the requirement | A - Q Q* A | < eps
TYPED_TEST(RandomizedRangeFinder, SingleShotApproximation) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);
  std::mt19937_64 randomEngine;
  randomEngine.seed(TestFixture::prngSeed);

  // Create a matrix with rank loss
  const MatrixType a = MatrixType::Random(TestFixture::numRows, TestFixture::dim) *
                       MatrixType::Random(TestFixture::dim, TestFixture::numCols);
  // Compute the randomized range approximation
  const MatrixType q = singleShot<MatrixType, std::mt19937_64>(
      a, TestFixture::dim + TestFixture::oversampling, randomEngine);
  const MatrixType res = a - q * (q.adjoint() * a);
  ASSERT_NEAR(res.norm(), 0, 2e3 * TestFixture::macheps);
}

// \brief Range approximation must have columns of unit length
TYPED_TEST(RandomizedRangeFinder, NoConditionerNorm) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);
  std::mt19937_64 randomEngine;
  randomEngine.seed(TestFixture::prngSeed);

  // Create a matrix with rank loss
  const MatrixType a = MatrixType::Random(TestFixture::numRows, TestFixture::dim) *
                       MatrixType::Random(TestFixture::dim, TestFixture::numCols);
  // Compute the randomized range approximation
  const MatrixType q =
      RandomizedSubspaceIterations<MatrixType, std::mt19937_64, NoConditioner>::compute(
          a, TestFixture::dim + TestFixture::oversampling, TestFixture::numIter, randomEngine);

  for (Index i = 0; i < TestFixture::dim; ++i) {
    ASSERT_NEAR(q.col(i).norm(), 1, 2 * TestFixture::macheps);
  }
}

// \brief Range approximation must have orthogonal columns
TYPED_TEST(RandomizedRangeFinder, NoConditionerOrthogonality) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);
  std::mt19937_64 randomEngine;
  randomEngine.seed(TestFixture::prngSeed);

  // Create a matrix with rank loss
  const MatrixType a = MatrixType::Random(TestFixture::numRows, TestFixture::dim) *
                       MatrixType::Random(TestFixture::dim, TestFixture::numCols);
  // Compute the randomized range approximation
  const MatrixType q =
      RandomizedSubspaceIterations<MatrixType, std::mt19937_64, NoConditioner>::compute(
          a, TestFixture::dim + TestFixture::oversampling, TestFixture::numIter, randomEngine);

  for (Index i = 0; i < TestFixture::dim; ++i) {
    for (Index j = 0; j < i; ++j) {
      const auto res = std::abs(q.col(i).dot(q.col(j)));
      ASSERT_NEAR(res, 0, 2 * TestFixture::macheps);
    }
  }
}

// \brief Range approximation must satisfy the requirement | A - Q Q* A | < eps
TYPED_TEST(RandomizedRangeFinder, NoConditionerApproximation) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);
  std::mt19937_64 randomEngine;
  randomEngine.seed(TestFixture::prngSeed);

  // Create a matrix with rank loss
  const MatrixType a = MatrixType::Random(TestFixture::numRows, TestFixture::dim) *
                       MatrixType::Random(TestFixture::dim, TestFixture::numCols);
  // Compute the randomized range approximation
  const MatrixType q =
      RandomizedSubspaceIterations<MatrixType, std::mt19937_64, NoConditioner>::compute(
          a, TestFixture::dim + TestFixture::oversampling, TestFixture::numIter, randomEngine);
  const MatrixType res = a - q * (q.adjoint() * a);
  ASSERT_NEAR(res.norm(), 0, 1e6 * TestFixture::macheps);
}

// \brief Range approximation must have columns of unit length
TYPED_TEST(RandomizedRangeFinder, LuConditionerNorm) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);
  std::mt19937_64 randomEngine;
  randomEngine.seed(TestFixture::prngSeed);

  // Create a matrix with rank loss
  const MatrixType a = MatrixType::Random(TestFixture::numRows, TestFixture::dim) *
                       MatrixType::Random(TestFixture::dim, TestFixture::numCols);
  // Compute the randomized range approximation
  const MatrixType q =
      RandomizedSubspaceIterations<MatrixType, std::mt19937_64, LuConditioner>::compute(
          a, TestFixture::dim + TestFixture::oversampling, TestFixture::numIter, randomEngine);

  for (Index i = 0; i < TestFixture::dim; ++i) {
    ASSERT_NEAR(q.col(i).norm(), 1, 2 * TestFixture::macheps);
  }
}

// \brief Range approximation must have orthogonal columns
TYPED_TEST(RandomizedRangeFinder, LuConditionerOrthogonality) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);
  std::mt19937_64 randomEngine;
  randomEngine.seed(TestFixture::prngSeed);

  // Create a matrix with rank loss
  const MatrixType a = MatrixType::Random(TestFixture::numRows, TestFixture::dim) *
                       MatrixType::Random(TestFixture::dim, TestFixture::numCols);
  // Compute the randomized range approximation
  const MatrixType q =
      RandomizedSubspaceIterations<MatrixType, std::mt19937_64, LuConditioner>::compute(
          a, TestFixture::dim + TestFixture::oversampling, TestFixture::numIter, randomEngine);

  for (Index i = 0; i < TestFixture::dim; ++i) {
    for (Index j = 0; j < i; ++j) {
      const auto res = std::abs(q.col(i).dot(q.col(j)));
      ASSERT_NEAR(res, 0, 2 * TestFixture::macheps);
    }
  }
}

// \brief Range approximation must satisfy the requirement | A - Q Q* A | < eps
TYPED_TEST(RandomizedRangeFinder, LuConditionerApproximation) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);
  std::mt19937_64 randomEngine;
  randomEngine.seed(TestFixture::prngSeed);

  // Create a matrix with rank loss
  const MatrixType a = MatrixType::Random(TestFixture::numRows, TestFixture::dim) *
                       MatrixType::Random(TestFixture::dim, TestFixture::numCols);
  // Compute the randomized range approximation
  const MatrixType q =
      RandomizedSubspaceIterations<MatrixType, std::mt19937_64, LuConditioner>::compute(
          a, TestFixture::dim + TestFixture::oversampling, TestFixture::numIter, randomEngine);
  const MatrixType res = a - q * (q.adjoint() * a);
  ASSERT_NEAR(res.norm(), 0, 4e2 * TestFixture::macheps);
}

// \brief Range approximation must have columns of unit length
TYPED_TEST(RandomizedRangeFinder, MgsConditionerNorm) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);
  std::mt19937_64 randomEngine;
  randomEngine.seed(TestFixture::prngSeed);

  // Create a matrix with rank loss
  const MatrixType a = MatrixType::Random(TestFixture::numRows, TestFixture::dim) *
                       MatrixType::Random(TestFixture::dim, TestFixture::numCols);
  // Compute the randomized range approximation
  const MatrixType q =
      RandomizedSubspaceIterations<MatrixType, std::mt19937_64, MgsConditioner>::compute(
          a, TestFixture::dim + TestFixture::oversampling, TestFixture::numIter, randomEngine);

  for (Index i = 0; i < TestFixture::dim; ++i) {
    ASSERT_NEAR(q.col(i).norm(), 1, 2 * TestFixture::macheps);
  }
}

// \brief Range approximation must have orthogonal columns
TYPED_TEST(RandomizedRangeFinder, MgsConditionerOrthogonality) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);
  std::mt19937_64 randomEngine;
  randomEngine.seed(TestFixture::prngSeed);

  // Create a matrix with rank loss
  const MatrixType a = MatrixType::Random(TestFixture::numRows, TestFixture::dim) *
                       MatrixType::Random(TestFixture::dim, TestFixture::numCols);
  // Compute the randomized range approximation
  const MatrixType q =
      RandomizedSubspaceIterations<MatrixType, std::mt19937_64, MgsConditioner>::compute(
          a, TestFixture::dim + TestFixture::oversampling, TestFixture::numIter, randomEngine);

  for (Index i = 0; i < TestFixture::dim; ++i) {
    for (Index j = 0; j < i; ++j) {
      const auto res = std::abs(q.col(i).dot(q.col(j)));
      ASSERT_NEAR(res, 0, 2 * TestFixture::macheps);
    }
  }
}

// \brief Range approximation must satisfy the requirement | A - Q Q* A | < eps
TYPED_TEST(RandomizedRangeFinder, MgsConditionerApproximation) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);
  std::mt19937_64 randomEngine;
  randomEngine.seed(TestFixture::prngSeed);

  // Create a matrix with rank loss
  const MatrixType a = MatrixType::Random(TestFixture::numRows, TestFixture::dim) *
                       MatrixType::Random(TestFixture::dim, TestFixture::numCols);
  // Compute the randomized range approximation
  const MatrixType q =
      RandomizedSubspaceIterations<MatrixType, std::mt19937_64, MgsConditioner>::compute(
          a, TestFixture::dim + TestFixture::oversampling, TestFixture::numIter, randomEngine);
  const MatrixType res = a - q * (q.adjoint() * a);
  ASSERT_NEAR(res.norm(), 0, 3e2 * TestFixture::macheps);
}

// \brief Range approximation must have columns of unit length
TYPED_TEST(RandomizedRangeFinder, QrConditionerNorm) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);
  std::mt19937_64 randomEngine;
  randomEngine.seed(TestFixture::prngSeed);

  // Create a matrix with rank loss
  const MatrixType a = MatrixType::Random(TestFixture::numRows, TestFixture::dim) *
                       MatrixType::Random(TestFixture::dim, TestFixture::numCols);
  // Compute the randomized range approximation
  const MatrixType q =
      RandomizedSubspaceIterations<MatrixType, std::mt19937_64, QrConditioner>::compute(
          a, TestFixture::dim + TestFixture::oversampling, TestFixture::numIter, randomEngine);

  for (Index i = 0; i < TestFixture::dim; ++i) {
    ASSERT_NEAR(q.col(i).norm(), 1, 2 * TestFixture::macheps);
  }
}

// \brief Range approximation must have orthogonal columns
TYPED_TEST(RandomizedRangeFinder, QrConditionerOrthogonality) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);
  std::mt19937_64 randomEngine;
  randomEngine.seed(TestFixture::prngSeed);

  // Create a matrix with rank loss
  const MatrixType a = MatrixType::Random(TestFixture::numRows, TestFixture::dim) *
                       MatrixType::Random(TestFixture::dim, TestFixture::numCols);
  // Compute the randomized range approximation
  const MatrixType q =
      RandomizedSubspaceIterations<MatrixType, std::mt19937_64, QrConditioner>::compute(
          a, TestFixture::dim + TestFixture::oversampling, TestFixture::numIter, randomEngine);

  for (Index i = 0; i < TestFixture::dim; ++i) {
    for (Index j = 0; j < i; ++j) {
      const auto res = std::abs(q.col(i).dot(q.col(j)));
      ASSERT_NEAR(res, 0, 2 * TestFixture::macheps);
    }
  }
}

// \brief Range approximation must satisfy the requirement | A - Q Q* A | < eps
TYPED_TEST(RandomizedRangeFinder, QrConditionerApproximation) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);
  std::mt19937_64 randomEngine;
  randomEngine.seed(TestFixture::prngSeed);

  // Create a matrix with rank loss
  const MatrixType a = MatrixType::Random(TestFixture::numRows, TestFixture::dim) *
                       MatrixType::Random(TestFixture::dim, TestFixture::numCols);
  // Compute the randomized range approximation
  const MatrixType q =
      RandomizedSubspaceIterations<MatrixType, std::mt19937_64, QrConditioner>::compute(
          a, TestFixture::dim + TestFixture::oversampling, TestFixture::numIter, randomEngine);
  const MatrixType res = a - q * (q.adjoint() * a);
  ASSERT_NEAR(res.norm(), 0, 3e2 * TestFixture::macheps);
}
