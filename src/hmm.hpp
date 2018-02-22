#ifndef HMM_HPP
#define HMM_HPP

#include <vector>

using state = int;

namespace ops {
  namespace hmm {
    
    using EigenMatrix = Eigen::Ref<const Eigen::MatrixXd>;

    /* abstract class */
    class HMM {
    public:
      HMM(std::size_t num_hidden, std::size_t num_obs)
	: m_num_hidden(n_hidden) {}

      auto log_likelihood() noexcept -> double {}

      double forward() noexcept {}
      double backward() noexcept {}
      auto baum_welch() noexcept {}

      std::vector<state>& infer(const std::vector<state>& states) noexcept {
	Eigen::MatrixXd<> temp1;
	Eigen::MatrixXd<> temp2;
	for (int i = 0; i < states.size(); i++) {
	  temp1(i,0) = m_initial_probs(i) * m_transition_probs(i, state[0]);
	  temp2(i,0) = 0;
	}
	for (int i = 0; i < m_num_hidden; i++) {
	  for (int j = 0; j < m_num_obs; j++) {
	    
	  }
	}
      }
      

    private:
      std::size_t m_num_hidden;
      std::size_t m_num_obs;

      // required, of dimensions [n_hidden, 1]
      Eigen::VectorXd m_initial_probs;
      Eigen::MatrixXd m_transition_probs;
      Eigen::MatrixXd m_emission_probs;
      Eigen::MatrixXd m_temporary;
    }; // class _HMM

  } // namespace hmm
} // namespace ops

#endif /* HMM_HPP */
