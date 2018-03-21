#ifndef HMM_HPP
#define HMM_HPP

#include <Eigen/Dense>
#include <vector>

using Matrix = Eigen::MatrixXd;
using MatrixType = Eigen::Ref<const Eigen::MatrixXd>;
using Vector = Eigen::VectorXd;
using VectorType = Eigen::Ref<const Eigen::VectorXd>;
using state = int;

template<std::size_t num_hidden, std::size_t num_obs>
class HMM {
public:
  HMM() = default;

private:
  Matrix<double, num_hidden, num_hidden> transition_probs;
  Matrix<double, num_hidden, num_obs> emission_probs;
  double init_prob = 1.0;
}

template<std::size_t num_obs, std::size_t num_hidden>
Vector forward(const MatrixType<num_hidden, num_hidden>& t_probs,
               const MatrixType<num_hidden, num_obs> e_probs,
               const VectorType& obs) {
  std::size_t num_obs = static_cast<std::size_t>obs.size();
  VectorType a = Vector::Zeros(num_obs);
  a(0) = 1.0;
  for (int t = 1; t < num_obs; ++t) {
    for (int st = 0; st < num_hidden; ++st) {
      
    }
  }
}

#endif /* HMM_HPP */
