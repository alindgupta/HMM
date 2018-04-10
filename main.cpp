#include "src/initializers.hpp"
#include "src/hmm.hpp"
#include <iostream>
#include <Eigen/Dense>

int main(int main, const char **argv) {

  // const int num_hidden = 3;
  // const int num_obs = 2;
  // const int obslen = 3;
  //
  // hmm::HMM x(num_hidden, num_obs);
  // 
  // Eigen::MatrixXd t(3,3);
  // t << 0.25, 0.75, 0, 0, 0.25, 0.75, 0, 0, 1.0;
  // x.transition_matrix(t);
  //
  // Eigen::MatrixXd e(3,2);
  // e << 1.0, 0.0, 0.0, 1.0, 1.0, 0.0;
  // x.emission_matrix(e);
  // 
  // Eigen::VectorXd o(3);
  // o << 1, 1, 0;
  //
  // auto a = x.infer(o);
  // std::cout << a << std::endl;

  Eigen::MatrixXd t(2,2);
  t << 0.7, 0.3, 0.4, 0.6;

  Eigen::MatrixXd e(2,3);
  e << 0.5, 0.4, 0.1, 0.1, 0.3, 0.6;

  Eigen::VectorXd i(2);
  i << 0.6, 0.4;

  hmm::HMM x(t, e, i);

  

  Eigen::VectorXd o(3);
  o << 0, 1, 2;
  auto a = x.viterbi(o);
  std::cout << a << std::endl;
  


}
