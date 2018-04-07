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
    
    /**
     * @brief Construct an HMM object 
     * given the sizes of hidden and observed states.
     *
     * Matrices of probabilities are initialized with the inverse of the
     * matrix size so that all states and emissions have equal probability.
     *
     * @param num_hidden Number N of hidden states.
     * @param num_observed Number K of observed states.
     */
    HMM(std::size_t num_hidden, std::size_t num_observed);

    /**
     * @brief Construct an HMM object with given probability matrices.
     *
     * Invokes copy constructors for the Eigen data types.
     *
     * @param a square matrix of transition probabilities (H x H)
     * @param a matrix of emission probabilities (H x K)
     */
    HMM(const MatrixType& transition_probs, const MatrixType&);

    /**
     * @brief Construct an HMM object with given probability matrices,
     * including a vector of initial probabilities.
     *
     * @param a square matrix of transition probabilities (H x H)
     * @param a matrix of emission probabilities (H x K)
     * @param a vector of initial probabilities (H x 1)
     *
     * Invokes copy constructors for the Eigen data types.
     */
    HMM(const MatrixType&, const MatrixType&, const VectorType&);

    /**
     * Calculate forward probabilities for a sequence of observed states.
     *
     * @param vector of observed states as ``int`` values
     * @returns matrix (H x N) of forward probabilities
     */
    Matrix forward(const VectorType&);

    
    Matrix backward(const VectorType&);

    /**
     * Calculate the Viterbi path for a sequence of observed states.
     * 
     * @param vector of observed states as ``int`` values
     * @returns a vector of the most likely sequence of hidden states
     *   that gave rise to the sequence of observed states provided as argument
     */ 
    Vector infer(const VectorType&);

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
