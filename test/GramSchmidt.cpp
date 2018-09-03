#include <complex>
#include <limits>

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <rsvd/GramSchmidt.hpp>

using Eigen::Index;
using Rsvd::Internal::modifiedGramSchmidt;

template <typename T> struct GramSchmidt : public ::testing::Test {
  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using RealType = typename Eigen::NumTraits<T>::Real;

  const unsigned int prngSeed = 777;

  const RealType macheps = std::numeric_limits<RealType>::epsilon();
};

using NumericalTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;

TYPED_TEST_CASE(GramSchmidt, NumericalTypes, );

/// \brief MGS function asserts that the number of columns is less or equal than the number of
/// columns. Otherwise, the columns are linearly dependent.
TYPED_TEST(GramSchmidt, MoreColumnsThanRows) {
  using MatrixType = typename TestFixture::MatrixType;

  MatrixType a = MatrixType::Zero(4, 6);
  ASSERT_DEATH(modifiedGramSchmidt(a),
               "Assertion `\\w+\\.cols\\(\\) <= \\w+\\.rows\\(\\)' failed");
}

/// \brief Test deflation on linearly dependent columns. All linearly-dependent columns must be
/// filled with zeros.
TYPED_TEST(GramSchmidt, LinearlyDependentColumnsEasy) {
  using MatrixType = typename TestFixture::MatrixType;

  MatrixType testee = MatrixType::Ones(10, 7);
  modifiedGramSchmidt(testee);

  MatrixType reference = MatrixType::Zero(10, 7);
  reference.col(0).setOnes();
  reference.col(0) /= sqrt(10);

  ASSERT_TRUE(testee.isApprox(reference));
}

/// \brief Test the deflation case when (new) independent directions come after linearly dependent
/// ones.
TYPED_TEST(GramSchmidt, LinearlyDependentColumnsOrder) {
  using MatrixType = typename TestFixture::MatrixType;

  MatrixType testee(5, 5);
  testee << 1, 1, 1, 1, 1, // 1
      0, 1, 1, 1, 1,       // 2
      0, 0, 0, 1, 1,       // 3
      0, 0, 0, 0, 1,       // 4
      0, 0, 0, 0, 0;       // 5
  modifiedGramSchmidt(testee);

  MatrixType reference(5, 5);
  reference << 1, 0, 0, 0, 0, // 1
      0, 1, 0, 0, 0,          // 2
      0, 0, 0, 1, 0,          // 3
      0, 0, 0, 0, 1,          // 4
      0, 0, 0, 0, 0;          // 5

  ASSERT_TRUE(testee.isApprox(reference));
}

/// \brief Test the deflation case when the columns are numerically linearly dependent.
TYPED_TEST(GramSchmidt, LinearlyDependentColumnsNumericalRankLoss) {
  using MatrixType = typename TestFixture::MatrixType;

  const auto e = 10 * TestFixture::macheps;

  MatrixType testee(5, 5);
  testee << 1, 1, 1, 1, 1, // 1
      0, 1, 1, 1, 1,       // 2
      0, 0, e, 1, 1,       // 3
      0, 0, 0, 0, 1,       // 4
      0, 0, 0, 0, 0;       // 5
  modifiedGramSchmidt(testee);

  MatrixType reference(5, 5);
  reference << 1, 0, 0, 0, 0, // 1
      0, 1, 0, 0, 0,          // 2
      0, 0, 0, 1, 0,          // 3
      0, 0, 0, 0, 1,          // 4
      0, 0, 0, 0, 0;          // 5

  ASSERT_TRUE(testee.isApprox(reference));
}

/// \brief A matrix which is already orthonormal must remain the same.
TYPED_TEST(GramSchmidt, AlreadyOrthogonalIdentity) {
  using MatrixType = typename TestFixture::MatrixType;

  const MatrixType reference = MatrixType::Identity(10, 4);
  MatrixType testee = MatrixType::Identity(10, 4);

  modifiedGramSchmidt(testee);

  ASSERT_TRUE(testee.isApprox(reference));
}

/// \brief A matrix which is already orthonormal must remain the same, larger matrix without
/// structure.
TYPED_TEST(GramSchmidt, AlreadyOrthogonalRandom) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);

  const Index numRows = 50;
  const Index numCols = 25;

  auto source = MatrixType::Random(numRows, numCols);
  Eigen::ColPivHouseholderQR<MatrixType> qr(source);
  const auto reference = qr.householderQ() * MatrixType::Identity(numRows, numCols);

  auto testee = reference;
  modifiedGramSchmidt(testee);

  ASSERT_TRUE(testee.isApprox(reference));
}

/// \brief Test if the matrix columns have unit norm after orthonormalization.
TYPED_TEST(GramSchmidt, CorrectNormalization) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);

  const Index numRows = 50;
  const Index numCols = 25;

  MatrixType testee = MatrixType::Random(numRows, numCols);
  modifiedGramSchmidt(testee);

  for (Index i = 0; i < numCols; ++i) {
    ASSERT_NEAR(testee.col(i).norm(), 1, 2 * TestFixture::macheps);
  }
}

/// \brief Test if the matrix columns are mutually orthogonal after orthonormalization.
TYPED_TEST(GramSchmidt, CorrectOrthogonality) {
  using MatrixType = typename TestFixture::MatrixType;

  std::srand(TestFixture::prngSeed);

  const Index numRows = 50;
  const Index numCols = 25;

  MatrixType testee = MatrixType::Random(numRows, numCols);
  modifiedGramSchmidt(testee);

  for (Index i = 0; i < numCols; ++i) {
    for (Index j = 0; j < i; ++j) {
      const auto res = std::abs(testee.col(i).dot(testee.col(j)));
      ASSERT_NEAR(res, 0, 2 * TestFixture::macheps);
    }
  }
}
