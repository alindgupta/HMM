#include "initializers.hpp"

using namespace hmm::initializers;

// replace multiple parameters with a "shape" parameter later
struct Shape {
public: 
  Shape(const std::size_t rows, const std::size_t cols)
    : rows(rows), cols(cols) {}
  
  Shape() = default;

  std::size_t rows = 1;
  std::size_t cols = 1;
};

struct ZeroInitializer : public Initializer {
  MatrixType operator()(const std::size_t rows,
                        const std::size_t cols) {
    return Matrix::Zero(rows, cols);
  }
};

struct ConstantInitializer : public Initializer {
  MatrixType operator()(const std::size_t rows,
                        const std::size_t cols,
                        const double x) {
    return Matrix::Constant(rows, cols, x);
  }
};

struct UniformInitializer : public Initializer {
  MatrixType operator()(const std::size_t rows,
                        const std::size_t cols,
                        const double lower,
                        const double upper) {
    auto runif = boost::random::uniform_real_distribution<>(lower, upper);

    // Eigen::Matrix::Zero returns an "expression object"
    // instead of an lvalue, so there is no unnecessary overhead to doing the following
    return Matrix::Zero(rows, cols)
      .unaryExpr([&](double t) { return runif(rng); });
  }
};

struct NormalInitializer : public Initializer {
  MatrixType operator()(const std::size_t rows,
                        const std::size_t cols,
                        const double mean,
                        const double var) {
    auto dnorm = boost::normal_distribution<>(mean, var);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<>>
      rnorm(rng, dnorm);
    return Matrix::Zero(rows, cols)
      .unaryExpr([&](double t){ return rnorm(); });
  }
};
