#ifndef RSVD_BENCHMARK_TYPES_ITERATOR_HPP_
#define RSVD_BENCHMARK_TYPES_ITERATOR_HPP_

#include <iostream>
#include <sstream>

#include "rsvd/Constants.hpp"

#include "BenchRunner.hpp"
#include "HelperTraits.hpp"

namespace Benchmark {

const char *tab = "  ";

template <typename T>
void benchHelper(const BenchConfig &, const RandomizedSvdConfig &, std::stringstream &) {}

template <typename T, typename Head, typename... Tail>
void benchHelper(const BenchConfig &benchConf, const RandomizedSvdConfig &rsvdConf,
                 std::stringstream &ss) {
  std::cout << "Using numerical type " << getMatrixTypeName<Head>() << std::endl;

  std::cout << tab << "Jacobi SVD" << std::endl;
  auto b1 = BenchRunnerSvd<Head>(benchConf);
  b1.run();
  std::cout << tab << tab;
  b1.displayResults();
  ss << "jacobi svd," << getMatrixTypeName<Head>() << ",";
  b1.pushAsCsv(ss);
  ss << std::endl;

  std::cout << tab << "Randomized SVD, no conditioner" << std::endl;
  auto b2 = BenchRunnerRandomizedSvd<Head, Rsvd::NoConditioner>(benchConf, rsvdConf);
  b2.run();
  std::cout << tab << tab;
  b2.displayResults();
  ss << "randomized svd : no conditioner," << getMatrixTypeName<Head>() << ",";
  b2.pushAsCsv(ss);
  ss << std::endl;

  std::cout << tab << "Randomized SVD, LU conditioner" << std::endl;
  auto b3 = BenchRunnerRandomizedSvd<Head, Rsvd::LuConditioner>(benchConf, rsvdConf);
  b3.run();
  std::cout << tab << tab;
  b3.displayResults();
  ss << "randomized svd : lu conditioner," << getMatrixTypeName<Head>() << ",";
  b3.pushAsCsv(ss);
  ss << std::endl;

  std::cout << tab << "Randomized SVD, MGS conditioner" << std::endl;
  auto b4 = BenchRunnerRandomizedSvd<Head, Rsvd::MgsConditioner>(benchConf, rsvdConf);
  b4.run();
  std::cout << tab << tab;
  b4.displayResults();
  ss << "randomized svd : mgs conditioner," << getMatrixTypeName<Head>() << ",";
  b4.pushAsCsv(ss);
  ss << std::endl;

  std::cout << tab << "Randomized SVD, QR conditioner" << std::endl;
  auto b5 = BenchRunnerRandomizedSvd<Head, Rsvd::QrConditioner>(benchConf, rsvdConf);
  b5.run();
  std::cout << tab << tab;
  b5.displayResults();
  ss << "randomized svd : qr conditioner," << getMatrixTypeName<Head>() << ",";
  b5.pushAsCsv(ss);
  ss << std::endl;

  benchHelper<T, Tail...>(benchConf, rsvdConf, ss);
}

template <typename... TypesList>
void benchIter(const BenchConfig &benchConf, const RandomizedSvdConfig &rsvdConf,
               std::stringstream &ss) {
  benchHelper<void, TypesList...>(benchConf, rsvdConf, ss);
}

} // namespace Benchmark

#endif
