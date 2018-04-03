#ifndef HMM_HPP
#define HMM_HPP

#include <Eigen/Dense>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using MatrixType = Eigen::Ref<const Eigen::MatrixXd>;
using VectorType = Eigen::Ref<const Eigen::VectorXd>;

namespace hmm {

  class HMM {
    
 zas public:
    HMM(std::size_t, std::size_t);
    HMM(const MatrixType&, const MatrixType&);
    
    Matrix forward(const VectorType&);
    Matrix backward(const VectorType&);
    void infer(const VectorType&);  // Viterbi algorithm

    // getters
    Matrix transition_matrix() const;
    Matrix emission_matrix() const;

    // setters
    void transition_matrix(const MatrixType&);
    void emission_matrix(const MatrixType&);
    
  private:
    std::size_t m_num_hidden = 1;
    std::size_t m_num_observed = 1;

    Matrix m_transition_probs =
      Matrix::Constant(m_num_hidden,
                       m_num_hidden,
                       1.0 / m_num_hidden);
    Matrix m_emission_probs =
      Matrix::Constant(m_num_hidden,
                       m_num_observed,
                       1.0 / m_num_observed);
    Vector m_initial_probs =
      Vector::Constant(m_num_hidden,
                       1.0 / m_num_observed);
  };
  
} /* class HMM */

#endif /* HMM_HPP */
