#include "hmm.hpp"

namespace hmm {

  HMM::HMM(std::size_t size_hidden, std::size_t size_obs)
    : m_num_hidden(size_hidden), m_num_observed(size_obs) {}

  HMM::HMM(const MatrixType& transition_matrix,
           const MatrixType& emission_matrix) {

    if (transition_matrix.cols() != transition_matrix.rows()) {
      throw "Transition matrix is not square.";
    }

    if (emission_matrix.rows() != transition_matrix.cols()) {
      throw "Dimension mismatch for emission and transition matrices.";
    }
    
    m_transition_probs = transition_matrix;
    m_emission_probs = emission_matrix;
    m_num_hidden = transition_matrix.rows();
    m_num_observed = emission_matrix.cols();
  }

  Matrix HMM::forward(const VectorType& obs) {
    std::size_t len_obs = obs.size();
    Matrix A = Matrix::Zero(m_num_hidden, len_obs);
    Vector temp(m_num_hidden);
    for (int s = 0; s < m_num_hidden; ++s) {
      temp = m_initial_probs(s) * m_emission_probs(s, obs(0)) * m_transition_probs.row(s);
      A.col(0) += temp;
    }
    for (int t = 1; t < len_obs; ++t) {
      for (int s = 0; s < m_num_hidden; ++s) {
        temp = A(s, t-1) * m_emission_probs(s, obs(t)) * m_transition_probs.row(s);
        A.col(t) += temp;
      }
    }
    Vector colsums = A.colwise().sum();
    for (int i = 0; i < A.cols(); ++i) {
      A.col(i) = A.col(i) / colsums(i);
    }
    return A;
  }

  Matrix HMM::backward(const VectorType& obs) {
    std::size_t len_obs = obs.size();
    Matrix A = Matrix::Zero(m_num_hidden, len_obs);
    Vector temp(m_num_hidden);

    A.col(0) = Vector::Constant(A.cols(), 1.0 / m_num_hidden);

    for (int t = 1; t < len_obs; ++t) {
      for (int s = 0; s < m_num_hidden; ++s) {
        temp = A(s, t-1) * m_emission_probs(s, obs(t)) * m_transition_probs.col(s);
        A.col(t) += temp;
      }
    }
    Vector colsums = A.colwise().sum();
    for (int i = 0; i < A.cols(); ++i) {
      A.col(i) = A.col(i) / colsums(i);
    }
    return A; 
  }

  Matrix HMM::transition_matrix(void) const {
    return m_transition_probs;
  }

  Matrix HMM::emission_matrix(void) const {
    return m_emission_probs;
  }

  void HMM::transition_matrix(const MatrixType& t) {
    m_transition_probs = t;
  }

  void HMM::emission_matrix(const MatrixType& e) {
    m_emission_probs = e;
  }
  
}  
