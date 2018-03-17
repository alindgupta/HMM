#include "src/initializers.hpp"
#include <iostream>
#include <cassert>

int main(int main, const char **argv) {

  auto c = NormalInitializer<10,10>(0.0, 0.1)();
  std::cout << c << std::endl;
  std::cout << c.rows() << "\n";
  
}
