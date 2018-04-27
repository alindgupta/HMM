#ifndef INITIALIZERS_HPP
#define INITIALIZERS_HPP

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <Eigen/Dense>

using Matrix = Eigen::MatrixXd;
using MatrixType = Eigen::Ref<const Eigen::MatrixXd>;

namespace hmm {
  namespace initializers {
    
    // random number generator
    static boost::mt19937 rng;

    /**
     * Base initializer class (virtual).
     * Subclasses should be implemented as functors
     * templated by parameters.
     * Calling the functor should return an Eigen matrix.
     *
     */
    struct Initializer {
    public:
      virtual MatrixType operator()() = 0;
    };


    /**
     * Initialize a matrix of zeros.
     * @tparam n_rows Number of rows
     * @tparam n_cols Number of columns
     *
     */
    template<std::size_t n_rows, std::size_t n_cols>
    struct ZeroInitializer : public Initializer {
    public:
      MatrixType operator()() override {
        return Matrix::Zero(n_rows, n_cols);
      }
    };

    
    /**
     * Initialize a matrix with uniformly distributed values.
     * @tparam n_rows Number of rows
     * @tparam n_cols Number of columns
     * @param lower Support lower bound for uniform distribution
     * @param upper Support upper bound for uniform distribution
     *
     */
    template<std::size_t n_rows, std::size_t n_cols>
    struct UniformInitializer : public Initializer {
    public:
      UniformInitializer(const double lower, const double upper)
        : m_lower(lower),
          m_upper(upper),
          m_unifd(boost::random::uniform_real_distribution<>(lower, upper)) {}

      MatrixType operator()() override {
        return Matrix::Zero(n_rows, n_cols)
          .unaryExpr([&](double t){ return m_unifd(rng); });
      }

    private:
      double m_lower;
      double m_upper;
      boost::random::uniform_real_distribution<double> m_unifd;
    };


    /**
     * Initialize a matrix with normally distributed values.
     * @tparam n_rows Number of rows
     * @tparam n_cols Number of columns
     * @param mean Location parameter for the normal distribution
     * @param var Variance parameter for the normal distribution
     *
     */
    template<std::size_t n_rows, std::size_t n_cols>
    struct NormalInitializer : public Initializer {
    public:
      NormalInitializer(const double mean, const double var)
        : m_mean(mean),
          m_var(var),
          m_gaussd(boost::normal_distribution<>(mean, var)) {}
      MatrixType operator()() override {
        boost::variate_generator<boost::mt19937&,
                                 boost::normal_distribution<>> gaussvars(rng, m_gaussd);
        return Matrix::Zero(n_rows, n_cols)
          .unaryExpr([&](double t){ return gaussvars(); });
      }

    private:
      double m_mean;
      double m_var;
      boost::normal_distribution<double> m_gaussd;
    };
  }
}

#endif
