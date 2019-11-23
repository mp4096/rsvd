#ifndef RSVD_BENCHMARK_DATA_STRUCTURES_HPP_
#define RSVD_BENCHMARK_DATA_STRUCTURES_HPP_

#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include <Eigen/Dense>

namespace Benchmark {

template <typename T> struct BenchStats {
  T relativeApproximationError{};
  std::chrono::milliseconds runtime{};

  void display() const {
    std::cout << "Runtime " << runtime.count()
              << " ms, relative approximation error: " << relativeApproximationError << std::endl;
  }

  void pushAsCsv(std::stringstream &ss) const {
    ss << std::setprecision(std::numeric_limits<T>::digits10) << relativeApproximationError << ","
       << runtime.count();
  }
};

struct RandomizedSvdConfig {
  Eigen::Index rank{};
  Eigen::Index oversampling{};
  unsigned int numIter{};
  unsigned int prngSeed{};

  void display() const {
    std::cout << "Randomized SVD rank " << rank << ", oversampling: " << oversampling
              << ", number of subspace iterations: " << numIter << ", PRNG seed: " << prngSeed
              << std::endl;
  }

  void pushAsCsv(std::stringstream &ss) const {
    ss << rank << "," << oversampling << "," << numIter << "," << prngSeed;
  }
};

struct BenchConfig {
  Eigen::Index numCols{};
  Eigen::Index numRows{};
  Eigen::Index rank{};
  unsigned int prngSeed{};

  void display() const {
    std::cout << "Test matrix shape " << numCols << " x " << numRows
              << ", test matrix rank: " << rank << ", PRNG seed: " << prngSeed << std::endl;
  }

  void pushAsCsv(std::stringstream &ss) const {
    ss << numCols << "," << numRows << "," << rank << "," << prngSeed;
  }
};

} // namespace Benchmark

#endif
