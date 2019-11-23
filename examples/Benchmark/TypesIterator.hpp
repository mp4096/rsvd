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

template <typename T>
void benchHelper(const BenchConfig &, const RandomizedSvdConfig &, std::stringstream &) {}

template <typename T, typename Head, typename... Tail>
void benchHelper(const BenchConfig &benchConf, const RandomizedSvdConfig &rsvdConf,
                 std::stringstream &ss) {
  std::cout << "Using numerical type " << getMatrixTypeName<Head>() << std::endl;

  {
    std::cout << kTab << "Jacobi SVD" << std::endl;
    auto b{BenchRunnerSvd<Head>(benchConf)};
    b.run();
    std::cout << kTab << kTab;
    b.displayResults();
    ss << "jacobi svd," << getMatrixTypeName<Head>() << ",";
    b.pushAsCsv(ss);
    ss << std::endl;
  }

  {
    std::cout << kTab << "Randomized SVD, no conditioner" << std::endl;
    auto b{BenchRunnerRandomizedSvd<Head, Rsvd::NoConditioner>(benchConf, rsvdConf)};
    b.run();
    std::cout << kTab << kTab;
    b.displayResults();
    ss << "randomized svd : no conditioner," << getMatrixTypeName<Head>() << ",";
    b.pushAsCsv(ss);
    ss << std::endl;
  }

  {
    std::cout << kTab << "Randomized SVD, LU conditioner" << std::endl;
    auto b{BenchRunnerRandomizedSvd<Head, Rsvd::LuConditioner>(benchConf, rsvdConf)};
    b.run();
    std::cout << kTab << kTab;
    b.displayResults();
    ss << "randomized svd : lu conditioner," << getMatrixTypeName<Head>() << ",";
    b.pushAsCsv(ss);
    ss << std::endl;
  }

  {
    std::cout << kTab << "Randomized SVD, MGS conditioner" << std::endl;
    auto b{BenchRunnerRandomizedSvd<Head, Rsvd::MgsConditioner>(benchConf, rsvdConf)};
    b.run();
    std::cout << kTab << kTab;
    b.displayResults();
    ss << "randomized svd : mgs conditioner," << getMatrixTypeName<Head>() << ",";
    b.pushAsCsv(ss);
    ss << std::endl;
  }

  {
    std::cout << kTab << "Randomized SVD, QR conditioner" << std::endl;
    auto b{BenchRunnerRandomizedSvd<Head, Rsvd::QrConditioner>(benchConf, rsvdConf)};
    b.run();
    std::cout << kTab << kTab;
    b.displayResults();
    ss << "randomized svd : qr conditioner," << getMatrixTypeName<Head>() << ",";
    b.pushAsCsv(ss);
    ss << std::endl;
  }

  benchHelper<T, Tail...>(benchConf, rsvdConf, ss);
}

template <typename... TypesList>
void benchIter(const BenchConfig &benchConf, const RandomizedSvdConfig &rsvdConf,
               std::stringstream &ss) {
  benchHelper<void, TypesList...>(benchConf, rsvdConf, ss);
}

} // namespace Benchmark

#endif
