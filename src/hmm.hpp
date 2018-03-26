#ifndef HMM_HPP
#define HMM_HPP

#include "initializers.hpp"
#include <Eigen/Dense>
#include <vector>

using Matrix = Eigen::MatrixXd;
using Vector= Eigen::VectorXd;
using MatrixType = Eigen::Ref<const Eigen::MatrixXd>;
using VectorType = Eigen::Ref<const Eigen::VectorXd>;
using state = int;

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
    HMM() = default;

    /**
     * Generate the forward probability of observing
     * a sequence of observations.
     * @param obs A vector of observed states
     *
     */
    Vector forward(const VectorType& obs) {
      
      // Begin by initializing an empty array that
      // will hold the forward probabilities
      auto a = ZeroInitializer<num_hidden, num_obs>()();

      Vector container(m_num_hidden);
      
      // We will calculate the forward probabilities
      // of the first state
      for (int h_i = 0; h_i < m_num_hidden; ++h_i) {
        container = m_init_probs(h_i) * m_emission_probs(obs(0), h_i) * m_transition_probs.row(h_i);
        a.col(0) = container.normalized();
      }

      // Now do the rest
      for (int t = 1; t < num_obs; ++t) {
        for (int h_i = 0; h_i < m_num_hidden; ++h_i) {
          container = a.col(t-1) * m_emission_probs(obs(t), h_i) * m_transition_probs.row(h_i);
          a.col(t) = container.normalized();
        }
      }

      return a;
    }

    auto tprobs() const {
      return m_transition_probs;
    }

    auto init_probs() const {
      return m_init_probs;
    }
    
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
    m_transition_probs = NormalInitializer<num_hidden, num_hidden>(0.0, 1.0)();

    /**
     * Emission probabilities 
     * This is a matrix of probabilities of emitting
     * an observable state o_i given the hidden state h_j
     * such that the sum of emitting any o_i given h_j is 1.
     */
    Matrix
    m_emission_probs =  NormalInitializer<num_hidden, num_obs>(0.0, 1.0)();

    Vector m_init_probs = UniformInitializer<num_hidden, 1>(0.09, 0.11)();

    

  };
} /* class HMM */

#endif /* HMM_HPP */
