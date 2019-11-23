#ifndef RSVD_BENCHMARK_TYPES_ITERATOR_HPP_
#define RSVD_BENCHMARK_TYPES_ITERATOR_HPP_

#include <iostream>
#include <sstream>

#include "rsvd/Constants.hpp"

#include "BenchRunner.hpp"
#include "HelperTraits.hpp"

namespace Benchmark {

namespace {
constexpr auto kTab{"  "};
}

template <typename MatrixType>
void jacobiSvdRun(const BenchConfig &benchConf, std::stringstream &ss) {
  std::cout << kTab << "Jacobi SVD" << std::endl;
  auto b{BenchRunnerSvd<MatrixType>(benchConf)};
  b.run();
  std::cout << kTab << kTab;
  b.displayResults();
  ss << "jacobi svd," << getMatrixTypeName<MatrixType>() << ",";
  b.pushAsCsv(ss);
  ss << std::endl;
}

template <typename MatrixType, Rsvd::SubspaceIterationConditioner Conditioner>
void randomizedSvdRun(const BenchConfig &benchConf, const RandomizedSvdConfig &rsvdConf,
                      std::stringstream &ss) {
  std::cout << kTab << "Randomized SVD, " << getConditionerName<Conditioner>() << std::endl;
  auto b{BenchRunnerRandomizedSvd<MatrixType, Conditioner>(benchConf, rsvdConf)};
  b.run();
  std::cout << kTab << kTab;
  b.displayResults();
  ss << "randomized svd : " << getConditionerName<Conditioner>() << ","
     << getMatrixTypeName<MatrixType>() << ",";
  b.pushAsCsv(ss);
  ss << std::endl;
}

template <typename T>
void benchHelper(const BenchConfig &, const RandomizedSvdConfig &, std::stringstream &) {}

template <typename T, typename Head, typename... Tail>
void benchHelper(const BenchConfig &benchConf, const RandomizedSvdConfig &rsvdConf,
                 std::stringstream &ss) {
  std::cout << "Using numerical type " << getMatrixTypeName<Head>() << std::endl;

  jacobiSvdRun<Head>(benchConf, ss);
  randomizedSvdRun<Head, Rsvd::SubspaceIterationConditioner::None>(benchConf, rsvdConf, ss);
  randomizedSvdRun<Head, Rsvd::SubspaceIterationConditioner::Lu>(benchConf, rsvdConf, ss);
  randomizedSvdRun<Head, Rsvd::SubspaceIterationConditioner::Mgs>(benchConf, rsvdConf, ss);
  randomizedSvdRun<Head, Rsvd::SubspaceIterationConditioner::Qr>(benchConf, rsvdConf, ss);

  benchHelper<T, Tail...>(benchConf, rsvdConf, ss);
}

template <typename... TypesList>
void benchIter(const BenchConfig &benchConf, const RandomizedSvdConfig &rsvdConf,
               std::stringstream &ss) {
  benchHelper<void, TypesList...>(benchConf, rsvdConf, ss);
}

} // namespace Benchmark

#endif
