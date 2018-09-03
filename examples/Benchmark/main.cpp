#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "DataStructures.hpp"
#include "TypesIterator.hpp"
#include <Eigen/Dense>

using Benchmark::BenchConfig;
using Benchmark::benchIter;
using Benchmark::RandomizedSvdConfig;
using Eigen::Index;
using Eigen::MatrixXcd;
using Eigen::MatrixXcf;
using Eigen::MatrixXd;
using Eigen::MatrixXf;

int main() {
  const unsigned int numRuns = 10;
  const unsigned int maxNumIter = 4;
  const auto sizes = std::vector<Index>{200, 500, 1000, 2000};

  std::stringstream resultsStream;
  resultsStream << "algo,"
                << "numtype,"
                << "num_cols,"
                << "num_rows,"
                << "rank,"
                << "prng_seed,"
                << "rsvd_rank,"
                << "oversampling,"
                << "num_subspace_iter,"
                << "rsvd_prng_seed,"
                << "error,"
                << "runtime" << std::endl;

  for (unsigned int runIdx = 0; runIdx < numRuns; ++runIdx) {
    std::cout << "Run " << runIdx + 1 << " out of " << numRuns << std::endl;
    for (unsigned int numIter = 0; numIter < (maxNumIter + 1); ++numIter) {
      for (Index size : sizes) {
        const BenchConfig benchConf = {
            size,         // numRows
            size / 2,     // numCols
            size / 4,     // rank
            777 + runIdx, // prngSeed
        };
        benchConf.display();

        const RandomizedSvdConfig rsvdConf = {
            size / 4,     // rank
            10,           // oversampling
            numIter,      // numIter
            777 + runIdx, // prngSeed
        };
        rsvdConf.display();

        benchIter<MatrixXf, MatrixXd, MatrixXcf, MatrixXcd>(benchConf, rsvdConf, resultsStream);
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  std::ofstream out("results.csv");
  out << resultsStream.str();
  out.close();

  return 0;
}
