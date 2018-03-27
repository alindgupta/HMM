#include "src/initializers.hpp"
#include "src/hmm.hpp"
#include <iostream>
#include <Eigen/Dense>

int main(int main, const char **argv) {

  constexpr int num_hidden = 3;
  constexpr int num_obs = 2;
  constexpr int obslen = 3;

  hmm::HMM<num_hidden, num_obs> x;

  Eigen::MatrixXd t(3,3);
  t << 0.25, 0.75, 0, 0, 0.25, 0.75, 0, 0, 1.0;
  x.set_tprobs(t);

  Eigen::MatrixXd e(3,2);
  e << 1.0, 0.0, 0.0, 1.0, 1.0, 0.0;
  x.set_eprobs(e);

  Eigen::VectorXd o(3);
  o << 0, 1, 0;

  auto a = x.forward(o);

  std::cout << a << std::endl;

}
