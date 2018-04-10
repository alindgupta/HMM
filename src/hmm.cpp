#include "hmm.hpp"
#include <iostream> // for debugging purposes, will be removed in the end

namespace hmm {

  /**
   * Construct an HMM object with random matrices.
   *
   * @param num_hidden Number N of hidden states.
   * @param num_observed Number K of observed states.
   *
   * Matrices of probabilities are initialized with the default value
   * of (1.0 / matrix size) so that all states and emissions 
   * have equal probability, and sum to 1.
   */
  HMM::HMM(std::size_t num_hidden, std::size_t num_obs) noexcept
    : m_num_hidden(num_hidden), m_num_observed(num_obs) {}

  
  /**
   * Construct an HMM object with given probability matrices.
   *
   * @param transition_matrix A square matrix of transition probabilities (H x H)
   * @param a matrix of emission probabilities (H x K)
   *
   * Invokes copy constructors for the Eigen data types.
   */

  HMM::HMM(const MatrixType& transition_matrix,
           const MatrixType& emission_matrix,
           const VectorType& initial_probabilities) {

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
    m_initial_probs = initial_probabilities;

    
  }

  /**
   * Implementation of the forward algorithm.
   *
   *
   *
   */
  Matrix HMM::forward(const VectorType& obs) noexcept {
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

  /**
   * Implementation of the backward algorithm.
   *
   *
   * @param obs Eigen Vector of observed sequences
   */
  Matrix HMM::backward(const VectorType& obs) noexcept {
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

  /**
   * Implementation of the Viterbi algorithm.
   * Given a sequence of observed states, infer the most likely
   * sequence of hidden states that gave rise to them.
   *
   * @param obs Eigen::Ref<const Eigen::VectorXd> (an Eigen Vector type) of
   *   observed states.
   */
  Vector HMM::infer(const VectorType& obs) noexcept {
    std::size_t len_obs = obs.size();
    Matrix A = Matrix::Zero(m_num_hidden, len_obs);
    Matrix B = Matrix::Zero(m_num_hidden, len_obs);
    for (int s = 0; s < m_num_hidden; ++s) {
      A(s,0) = m_initial_probs(s) * m_emission_probs(s, obs(0));
      B(s,0) = 0.0;
    }

    Vector tmp(len_obs);
    Vector::Index argmax;
    for (int t = 1; t < len_obs; ++t) {
      for (int s = 0; s < m_num_hidden; ++s) {
        tmp =  m_emission_probs.col(obs(t-1)).array() * A.col(t-1).array() * m_transition_probs.col(s).array();

        A(s, t) = tmp.maxCoeff(&argmax);
        B(s, t) = argmax;
      }
    }


    Vector Z(len_obs);
    auto i = A.col(len_obs-1).maxCoeff(&argmax);
    Z(len_obs-1) = argmax;
    Vector X(len_obs);
    X(len_obs-1) = Z(len_obs-1);
    for (int t = len_obs-1; t > 0; --t) {
      Z(t-1) = B(Z(t),t);
      X(t-1) = Z(t-1);
    }
    return X;
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
