/*
 *       Usage
 *       >> UniformInitializer<2,2> m(1.0, 2.0);
 *       >> auto p = Matrix(m);
 */

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <Eigen/Dense>


namespace ops {

  static boost::mt19937 rng;
  using MatrixXdType = Eigen::Ref<const Eigen::MatrixXd>;

   /*
   *  Base initializer struct (abstract)
   *
   **/
  struct Initializer {
  public:
    virtual MatrixXdType operator()() = 0;
  };

  /*
  * Zero initializer
  * Templated on the number of rows and columns
  * Returns Eigen matrix containing all zeros
  **/
  template<std::size_t n_rows, std::size_t n_cols>
  struct ZeroInitializer : public Initializer {
  public:
    MatrixXdType operator()() override {
      return Eigen::MatrixXd::Zero(n_rows, n_cols);
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
      return Eigen::MatrixXd::Zero(n_rows, n_cols)
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
      return Eigen::MatrixXd::Zero(n_rows, n_cols)
        .unaryExpr([&](double t){ return gaussvars(); });
    }

  private:
    double m_mean;
    double m_var;
    boost::normal_distribution<double> m_gaussd;
  };


  /*
  *
  *
  *
  **/
  class Matrix : public Eigen::MatrixXd {
  public:
    template<typename T>
    explicit Matrix(T callable) {
      m_mat = callable();
      m_ncols = m_mat.cols();
      m_nrows = m_mat.rows();
    }

    auto diagonal() { return m_mat.diagonal(); }

    auto mat() { return m_mat; }

    inline std::size_t nrows() const { return m_nrows; }

    inline std::size_t ncols() const { return m_ncols; }

  private:
    std::size_t m_nrows = 1;
    std::size_t m_ncols = 1;
    Eigen::MatrixXd m_mat;  // type can not be MatrixXdType
  };

} // namespace ops

#endif /* MATRIX_HPP */
