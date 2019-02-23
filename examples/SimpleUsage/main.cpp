#include <iostream>
#include <random>
#include <rsvd/Prelude.hpp>

using Eigen::ComputeThinU;
using Eigen::ComputeThinV;
using Eigen::Index;
using Eigen::JacobiSVD;
using Eigen::MatrixXd;

int main() {
  const Index numRows = 350;
  const Index numCols = 240;
  const Index reducedRank = 50;

  // Initialize PRNG for the Eigen random matrix generation
  std::srand((unsigned int)777);
  const MatrixXd x = MatrixXd::Random(numRows, numCols);

  // Jacobi SVD
  JacobiSVD<MatrixXd> svd(x, ComputeThinU | ComputeThinV);
  const MatrixXd svdApprox = svd.matrixU().leftCols(reducedRank) *
                             svd.singularValues().head(reducedRank).asDiagonal() *
                             svd.matrixV().leftCols(reducedRank).adjoint();
  const auto svdErr = Rsvd::relativeFrobeniusNormError(x, svdApprox);
  std::cout << "SVD reconstruction error: " << svdErr << std::endl;

  // Randomized SVD
  std::mt19937_64 randomEngine;
  randomEngine.seed(777);

  Rsvd::RandomizedSvd<MatrixXd, std::mt19937_64, Rsvd::LuConditioner> rsvd(randomEngine);
  rsvd.compute(x, reducedRank);
  const MatrixXd rsvdApprox =
      rsvd.matrixU() * rsvd.singularValues().asDiagonal() * rsvd.matrixV().adjoint();
  const auto rsvdErr = Rsvd::relativeFrobeniusNormError(x, rsvdApprox);
  std::cout << "Randomized SVD reconstruction error: " << rsvdErr << std::endl;

  return 0;
}
