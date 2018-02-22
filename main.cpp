#include "src/matrix.hpp"
#include <iostream>
#include <cassert>

/* IO */
int main(int main, const char **argv) {

  using namespace ops;

  auto a = NormalInitializer<500,500>(0.0, 0.1);
  Matrix b = Matrix(a);
  auto c = b.mat();
  
  
}
