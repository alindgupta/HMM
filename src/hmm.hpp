#ifndef HMM_HPP
#define HMM_HPP

#include "initializers.hpp"
#include <Eigen/Dense>
#include <vector>

using Matrix = Eigen::MatrixXd;
using Vector= Eigen::VectorXd;
using MatrixType = Eigen::Ref<const Eigen::MatrixXd>;
using VectorType = Eigen::Ref<const Eigen::VectorXd>;

using namespace hmm::initializers;

namespace hmm {
  
  /**
   * Instantiate an HMM object.
   * @tparam num_hidden Dimensionality of the hidden state
   * @tparam n_obs Dimensionality of observable state
   *
   */
  template<std::size_t num_hidden, std::size_t num_obs>
  class HMM {
    
  public:
    
    /**
     * Default constructor for HMM.
     *
     */
    HMM() = default;

    /**
     * Generate the forward probability of observing
     * a sequence of observations.
     * @param obs A vector of observed states
     *
     */
    Matrix forward(const VectorType& obs) {
      std::size_t len_obs = obs.size();

      // Initialize a matrix to hold forward probabilities
      Matrix a = Matrix::Zero(num_hidden, len_obs);
      Vector container(num_hidden);
      
      // Calculate forward probabilities for the first observation
      for (int h_i = 0; h_i < num_hidden; ++h_i) {
        container = m_initial_probs(h_i)
          * m_emission_probs(h_i, obs(0))
          * m_transition_probs.row(h_i);
        a.col(0) += container;
      }

      // Calculate forward probabilities for the rest of the observations
      for (int t = 1; t < len_obs; ++t) {
        for (int h_i = 0; h_i < num_hidden; ++h_i) {
          container = a(h_i, t-1)
            * m_emission_probs(h_i, obs(t))
            * m_transition_probs.row(h_i);
          a.col(t) += container;
        }
      }

      // Normalize forward probabilities by column
      Vector colsums = a.colwise().sum();
      for (int i = 0; i < a.cols(); ++i) {
        a.col(i) = (a.col(i) / colsums(i)).transpose();
      }
  
      return a;
    }

    // getters and setters
    auto tprobs() const { return m_transition_probs; }
    auto eprobs() const { return m_emission_probs; }
    auto iprobs() const { return m_initial_probs; }
    void set_tprobs(const Matrix& m) { m_transition_probs = m; }
    void set_eprobs(const Matrix& m) { m_emission_probs = m; }
    void set_iprobs(const Vector& v) { m_initial_probs = v; }
    
    
  private:
    int m_num_hidden = num_hidden;
    int m_num_obs = num_obs;

    /**
     * Transition probabilities 
     * This is a symmetric matrix indexed by state
     * such that the sum of transition probabilities
     * from state i to state j is 1.
     * The rows signify "from" state and columns signify "to" state
     * where the sum over rows is 1.
     */
    Matrix
    m_transition_probs = Matrix::Constant(num_hidden,
                                          num_hidden,
                                          1.0 / num_hidden);

    /**
     * Emission probabilities 
     * This is a matrix of probabilities of emitting
     * an observable state o_i given the hidden state h_j
     * such that the sum of emitting any o_i given h_j is 1.
     */
    Matrix
    m_emission_probs =  Matrix::Constant(num_hidden,
                                         num_obs,
                                         1.0 / num_obs);

    Vector
    m_initial_probs = Vector::Constant(num_hidden,
                                    1.0 / num_hidden);

  };
} /* class HMM */

#endif /* HMM_HPP */
