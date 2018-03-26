#include "src/initializers.hpp"
#include "src/hmm.hpp"
#include <iostream>
#include <Eigen/Dense>

int main(int main, const char **argv) {

  // auto x = hmm::initializers::NormalInitializer<10,10>(0.0, 0.1)();
  hmm::HMM<10, 5> y;
  Eigen::VectorXd p(5);
  p << 1, 2, 3, 1, 1;
  // y.forward(p);
}
