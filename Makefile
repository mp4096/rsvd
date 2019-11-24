.DEFAULT_GOAL := help
ifeq ($(TOOLCHAIN),)
TOOLCHAIN := CC=clang-8 CXX=clang++-8
endif

eigen_version := 3.3.5

.PHONY: benchmark
benchmark: ## Compile the benchmark
	mkdir -p build
	cd build && $(TOOLCHAIN) cmake ..
	cd build && make -j2 example_benchmark

.PHONY: benchmark_release
benchmark_release: clean ## Compile (release) the benchmark
	mkdir -p build
	cd build && $(TOOLCHAIN) cmake -DCMAKE_BUILD_TYPE=Release ..
	cd build && make -j2 example_benchmark

.PHONY: simple_usage
simple_usage: ## Compile the simple usage example
	mkdir -p build
	cd build && $(TOOLCHAIN) cmake ..
	cd build && make -j2 example_simple_usage

.PHONY: simple_usage_release
simple_usage_release: clean ## Compile (release) the simple usage example
	mkdir -p build
	cd build && $(TOOLCHAIN) cmake -DCMAKE_BUILD_TYPE=Release ..
	cd build && make -j2 example_simple_usage

.PHONY: unit_tests
unit_tests: ## Run unit tests
	mkdir -p build
	cd build && $(TOOLCHAIN) cmake .. && make -j2 rsvd_test
	./build/rsvd_test

.PHONY: clean
clean: ## Purge all build artifacts
	rm -rf "./build"

.PHONY: doxygen
doxygen: ## Generate doxygen documentation
	mkdir -p doc
	doxygen Doxyfile

.PHONY: install_eigen
install_eigen: ## Install Eigen from source
	wget https://bitbucket.org/eigen/eigen/get/$(eigen_version).tar.bz2 -O /tmp/eigen.tar.bz2
	mkdir -p eigen3 && tar -xvjf /tmp/eigen.tar.bz2 -C eigen3 --strip-components 1

.PHONY: fmt
fmt: ## Pretty print source code
	clang-format-6.0 -i include/rsvd/*.hpp
	clang-format-6.0 -i test/*.cpp
	clang-format-6.0 -i examples/Benchmark/*.cpp
	clang-format-6.0 -i examples/Benchmark/*.hpp
	clang-format-6.0 -i examples/SimpleUsage/*.cpp

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'
