#ifndef HMM_HPP
#define HMM_HPP

#include <Eigen/Dense>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using MatrixType = Eigen::Ref<const Eigen::MatrixXd>;
using VectorType = Eigen::Ref<const Eigen::VectorXd>;


namespace hmm {

  class HMM {
    
  public:

    HMM() = default;
    
    /**
     * @brief Construct an HMM object 
     * given the sizes of hidden and observed states.
     *
     * Matrices of probabilities are initialized with the inverse of the
     * matrix size so that all states and emissions have equal probability.
     *
     * @param num_hidden Number of hidden states.
     * @param num_observed Number of observed states.
     */
    HMM(std::size_t num_hidden, std::size_t num_observed) noexcept;

    /**
     * @brief Construct an HMM object with given probability matrices.
     *
     * Invokes copy constructors for the Eigen data types.
     *
     * @param transition_matrix Matrix of transition probabilities of
     * dimensions [num_hidden, num_hidden].
     *
     * @param emission_matrix Matrix of emission probabilities of
     * dimensions [num_hidden, num_observed].
     *
     * @param initial_probs Vector of initial probabilities.
     */
    HMM(const MatrixType& transition_probs,
        const MatrixType& emission_probs,
        const VectorType& initial_probs);

    HMM(const MatrixType& transition_probs,
        const MatrixType& emission_probs);

    /**
     * @brief Calculate forward probabilities for a sequence of observed states.
     *
     * @param observations Vector of observed states as `int` values.
     * @returns Matrix of forward probabilities.
     */
    Matrix forward(const VectorType& observations) noexcept;

    
    Matrix backward(const VectorType&) noexcept;

    /**
     * Calculate the Viterbi path for a sequence of observed states.
     * 
     * @param Vector of observed states as `int` values.
     *
     * @return Vector of the most likely sequence of hidden states
     * that gave rise to the sequence of observed states provided as argument.
     */ 
    Vector viterbi(const VectorType&) noexcept;

    // getters
    Matrix transition_matrix() const noexcept;
    Matrix emission_matrix() const noexcept;

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
