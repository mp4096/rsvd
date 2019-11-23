#ifndef RSVD_BENCHMARK_BENCH_RUNNER_HPP_
#define RSVD_BENCHMARK_BENCH_RUNNER_HPP_

#include <chrono>
#include <cstdlib>
#include <rsvd/Prelude.hpp>
#include <sstream>

#include "DataStructures.hpp"
#include <Eigen/Dense>

namespace Benchmark {

template <typename MatrixType> MatrixType generateTestMatrix(const BenchConfig &conf) {
  std::srand(conf.prngSeed);

  const auto a = MatrixType::Random(conf.numRows, conf.rank);
  const auto b = MatrixType::Random(conf.rank, conf.numCols);

  return a * b;
}

template <typename MatrixType> class BenchRunnerBase {
  using ScalarType = typename MatrixType::Scalar;
  using RealType = typename Eigen::NumTraits<ScalarType>::Real;

public:
  explicit BenchRunnerBase(const BenchConfig &benchConf)
      : m_testMatrix(generateTestMatrix<MatrixType>(benchConf)), m_benchConfig(benchConf),
        m_stats() {}

  virtual void run() = 0;

  virtual void pushAsCsv(std::stringstream &) const = 0;

  void displayResults() const { m_stats.display(); }

protected:
  const MatrixType m_testMatrix;
  const BenchConfig &m_benchConfig;
  BenchStats<RealType> m_stats;
};

template <typename MatrixType> class BenchRunnerSvd : public BenchRunnerBase<MatrixType> {
  using ScalarType = typename MatrixType::Scalar;
  using RealType = typename Eigen::NumTraits<ScalarType>::Real;

public:
  explicit BenchRunnerSvd(const BenchConfig &benchConf) : BenchRunnerBase<MatrixType>(benchConf) {}

  void run() {
    const auto tic = std::chrono::steady_clock::now();
    Eigen::JacobiSVD<MatrixType> svd(this->m_testMatrix,
                                     Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto toc = std::chrono::steady_clock::now();

    this->m_stats.runtime = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);

    const MatrixType reconstructed =
        svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().adjoint();

    this->m_stats.relativeApproximationError =
        Rsvd::relativeFrobeniusNormError(this->m_testMatrix, reconstructed);
  }

  void pushAsCsv(std::stringstream &ss) const {
    this->m_benchConfig.pushAsCsv(ss);
    ss << ",,,,,";
    this->m_stats.pushAsCsv(ss);
  }
};

template <typename MatrixType, Rsvd::SubspaceIterationConditioner Conditioner>
class BenchRunnerRandomizedSvd : public BenchRunnerBase<MatrixType> {
  using ScalarType = typename MatrixType::Scalar;
  using RealType = typename Eigen::NumTraits<ScalarType>::Real;

public:
  BenchRunnerRandomizedSvd(const BenchConfig &benchConf, const RandomizedSvdConfig &rsvdConf)
      : BenchRunnerBase<MatrixType>(benchConf), m_rsvdConfig(rsvdConf) {}

  void run() {
    std::mt19937_64 randomEngine;
    randomEngine.seed(m_rsvdConfig.prngSeed);

    const auto tic = std::chrono::steady_clock::now();
    Rsvd::RandomizedSvd<MatrixType, std::mt19937_64, Conditioner> rsvd(randomEngine);
    rsvd.compute(this->m_testMatrix, m_rsvdConfig.rank, m_rsvdConfig.oversampling,
                 m_rsvdConfig.numIter);
    const auto toc = std::chrono::steady_clock::now();

    this->m_stats.runtime = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);

    const MatrixType reconstructed =
        rsvd.matrixU() * rsvd.singularValues().asDiagonal() * rsvd.matrixV().adjoint();

    this->m_stats.relativeApproximationError =
        Rsvd::relativeFrobeniusNormError(this->m_testMatrix, reconstructed);
  }

  void pushAsCsv(std::stringstream &ss) const {
    this->m_benchConfig.pushAsCsv(ss);
    ss << ",";
    this->m_rsvdConfig.pushAsCsv(ss);
    ss << ",";
    this->m_stats.pushAsCsv(ss);
  }

private:
  const RandomizedSvdConfig &m_rsvdConfig;
};

} // namespace Benchmark

#endif
