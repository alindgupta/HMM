#ifndef INITIALIZERS_HPP
#define INITIALIZERS_HPP

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <Eigen/Dense>

using MatrixXd = Eigen::MatrixXd;
using MatrixXdType = Eigen::Ref<const Eigen::MatrixXd>;

// random number generator
static boost::mt19937 rng;

struct Initializer {
public:
  virtual MatrixXdType operator()() = 0;
};

template<std::size_t n_rows, std::size_t n_cols>
struct ZeroInitializer : public Initializer {
public:
  MatrixXdType operator()() override {
    return MatrixXd::Zero(n_rows, n_cols);
  }
};

template<std::size_t n_rows, std::size_t n_cols>
struct UniformInitializer : public Initializer {
public:
  UniformInitializer(const double lower, const double upper)
    : m_lower(lower),
      m_upper(upper),
      m_unifd(boost::random::uniform_real_distribution<>(lower, upper)) {}

  MatrixXdType operator()() override {
    return MatrixXd::Zero(n_rows, n_cols)
      .unaryExpr([&](double t){ return m_unifd(rng); });
  }

private:
  double m_lower;
  double m_upper;
  boost::random::uniform_real_distribution<double> m_unifd;
};

template<std::size_t n_rows, std::size_t n_cols>
struct NormalInitializer : public Initializer {
public:
  NormalInitializer(const double mean, const double var)
    : m_mean(mean),
      m_var(var),
      m_gaussd(boost::normal_distribution<>(mean, var)) {}
  MatrixXdType operator()() override {
    boost::variate_generator<boost::mt19937&,
                             boost::normal_distribution<>> gaussvars(rng, m_gaussd);
    return MatrixXd::Zero(n_rows, n_cols)
      .unaryExpr([&](double t){ return gaussvars(); });
  }

private:
  double m_mean;
  double m_var;
  boost::normal_distribution<double> m_gaussd;
};

#endif
