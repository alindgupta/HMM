#include "hmm.hpp"
#include <iostream> // for debugging purposes, will be removed in the end

namespace hmm {

  /**
   * @brief Construct an HMM object with random matrices.
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
   * @brief Construct an HMM object with given probability matrices.
   *
   * Invokes copy constructors for the Eigen data types.
   *
   * @param transition_matrix A square matrix of transition probabilities.
   * Rows represent "from" state and columns represent "to" state.
   * Value at row i, column j is the probability of transitioning from i to j.
   *
   * @param emission_matrix A matrix of emission probabilities.
   * Rows represent hidden states.
   * Columns represent observed states.
   * 
   * @param initial_probabilities A vector of initial probabilities.
   * Length corresponds to number of hidden states.
   *
   * @throws If matrix dimensions are mistmatched.
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
    
    m_num_hidden = transition_matrix.rows();
    m_num_observed = emission_matrix.cols();
    
    m_transition_probs = transition_matrix;
    m_emission_probs = emission_matrix;
    m_initial_probs = initial_probabilities;
  }
  

  /**
   * @brief Implementation of the forward algorithm.
   *
   * @param obs Vector of observed sequences.
   * @return Matrix of forward probabilities.
   */
  Matrix HMM::forward(const VectorType& obs) noexcept {
    std::size_t len_obs = obs.size();

    // Initialize a matrix to hold forward probabilities
    Matrix A = Matrix(m_num_hidden, len_obs);

    // Initialize a vector to hold (temporary) products
    Vector prod(m_num_hidden);
    
    // Calculate forward probability of first observation
    for (int s = 0; s < m_num_hidden; ++s) {
      prod = m_initial_probs(s)
        * m_emission_probs(s, obs(0))
        * m_transition_probs.row(s);
      A.col(0) += prod;
    }

    // Calculate forward probabilities for the rest of the observations
    for (int t = 1; t < len_obs; ++t) {
      for (int s = 0; s < m_num_hidden; ++s) {
        prod = A(s, t-1)
          * m_emission_probs(s, obs(t))
          * m_transition_probs.row(s);
        A.col(t) += prod;
      }
    }
    
    // Normalize forward probabilities between 0 and 1
    Vector colsums = A.colwise().sum();
    for (int i = 0; i < A.cols(); ++i) {
      A.col(i) = A.col(i) / colsums(i);
    }
    
    return A;
  }

  
  /**
   * @brief Implementation of the backward algorithm.
   *
   * @param obs Vector of observed sequences.
   * @return Matrix of backward probabilities.
   */
  Matrix HMM::backward(const VectorType& obs) noexcept {
    std::size_t len_obs = obs.size();

    // Initialize a matrix to hold forward probabilities
    Matrix A = Matrix::Zero(m_num_hidden, len_obs);

    // Initialize a vector to hold (temporary) products
    Vector prod(m_num_hidden);

    A.col(0) = Vector::Constant(A.cols(), 1.0 / m_num_hidden);

    for (int t = 1; t < len_obs; ++t) {
      for (int s = 0; s < m_num_hidden; ++s) {
        prod = A(s, t-1)
          * m_emission_probs(s, obs(t))
          * m_transition_probs.col(s);
        A.col(t) += prod;
      }
    }

    // Normalize forward probabilities between 0 and 1
    Vector colsums = A.colwise().sum();
    for (int i = 0; i < A.cols(); ++i) {
      A.col(i) = A.col(i) / colsums(i);
    }
    return A; 
  }

  
  /**
   * @brief Implementation of the Viterbi algorithm.
   * Given a sequence of observed states, infer the most likely
   * sequence of hidden states that gave rise to them.
   *
   * @param obs Vector of observed states.
   * @return Vector of the Viterbi path of most likely states.
   */
  Vector HMM::viterbi(const VectorType& obs) noexcept {
    std::size_t obs_len = obs.size();

    // Initialize containers of zeros to store probabilities
    Matrix A = Matrix(m_num_hidden, obs_len);
    Matrix B = Matrix(m_num_hidden, obs_len);

    // Calculate probabilities for the first observation
    for (int s = 0; s < m_num_hidden; ++s) {
      A(s,0) = m_initial_probs(s) * m_emission_probs(s, obs(0));
      B(s,0) = 0.0;
    }

    Vector prod(m_num_hidden); // product of probabilities on the trellis
    Vector::Index argmax;

    // Calculate probabilities for the rest of the observations
    for (int t = 1; t < obs_len; ++t) {
      for (int s = 0; s < m_num_hidden; ++s) {
        prod =  m_emission_probs(s, obs(t))
          * A.col(t-1).array()
          * m_transition_probs.col(s).array();
        A(s, t) = prod.maxCoeff(&argmax);
        B(s, t) = argmax;
      }
    }
    
    // Initialize a vector to hold the Viterbi path
    Vector viterbi_path(obs_len);
    
    A.col(obs_len-1).maxCoeff(&argmax);
    viterbi_path(obs_len-1) = argmax;

    // Calculate the Viterbi path
    for (int t = obs_len-1; t > 0; --t) {
      viterbi_path(t-1) = B(viterbi_path(t),t);
    }
    
    return viterbi_path;
  }

  
  /**
   * Getter for transition probabilities.
   */
  Matrix HMM::transition_matrix(void) const noexcept {
    return m_transition_probs;
  }

  
  /**
   * Getter for emission probabilities.
   */
  Matrix HMM::emission_matrix(void) const noexcept {
    return m_emission_probs;
  }

  
  /**
   * Setter for transition probabilities.
   */
  void HMM::transition_matrix(const MatrixType& t) {
    
    m_transition_probs = t;
  }

  
  /**
   * Setter for emission probabilities.
   */
  void HMM::emission_matrix(const MatrixType& e) {
    m_emission_probs = e;
  }
  
} // namespace hmm 
