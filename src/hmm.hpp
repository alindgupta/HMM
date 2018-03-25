#ifndef HMM_HPP
#define HMM_HPP

#include <Eigen/Dense>
#include <vector>

using Matrix = Eigen::MatrixXd;
using Vector= Eigen::VectorXd;
using MatrixType = Eigen::Ref<const Eigen::MatrixXd>;
using VectorType = Eigen::Ref<const Eigen::VectorXd>;
using state = int;

using namespace initializers;

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

      std::size_t num_obs = static_cast<std::size_t>obs.size();
      VectorType a = Vector::Zeros(num_obs);
      a(0) = 1.0;

      for (int t = 1; t < num_obs; ++t) {
        auto e_obs = m_emission_probs(obs(t));
        for (int h = 0; h < num_hidden; ++h) {
          auto a_t0 = a(t-1);
          auto t_t0 = m_transition_probs(/* something */);
          auto out = a_t0 * t_t0;                               
        }
        a(t) = e_obs * out.sum();
      }
    
    private:
      int m_num_hidden = num_hidden;
      int m_num_obs = num_obs;

      /**
       * Transition probabilities 
       * This is a symmetric matrix indexed by state
       * such that the sum of transition probabilities
       * from state i to state j is 1.
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

    }
  } /* class HMM */

#endif /* HMM_HPP */
