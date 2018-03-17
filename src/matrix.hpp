#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "initializers.hpp"

using MatrixXd = Eigen::MatrixXd;

class Matrix {
public:
  Matrix(MatrixXd&& temp) {
    m_num_rows = temp.rows();
    m_num_cols = temp.cols();
    m_matrix = std::move(temp);
  }
  
  template<typename Func>
  Matrix(Func f) {}
    
  auto diagonal() { return m_matrix.diagonal(); }
    
  auto matrix() { return m_matrix; }
    
  inline std::size_t rows() const { return m_num_rows; }
    
  inline std::size_t cols() const { return m_num_cols; }
    
private:
  std::size_t m_num_rows = 1;
  std::size_t m_num_cols = 1;
  MatrixXd m_matrix;
};


#endif /* MATRIX_HPP */
