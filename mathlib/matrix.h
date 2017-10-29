/*
    Object "Matrix" for linear algebra.
    (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_MATRIX_H
#define MATHLIB_MATRIX_H

#include <vector>
#include <algorithm>
#include <type_traits>

namespace mathlib {

template <typename T>
class matrix
{
  static_assert(std::is_arithmetic<T>::value == true, "Matrix supports only an arithmetic type.");

  using data_container = std::vector<typename T>;
  using const_iterator = typename data_container::const_iterator;
  using iterator = typename data_container::iterator;

  template <typename iter_t>
  class row {
  public:
    row() = delete;
    row(row&& /*r*/) = default;
    row(const row& /*r*/) = default;
    row(iter_t&& r) : row_begin(r) {}
    typename iter_t::reference operator [] (size_t c) {
      return *(row_begin + c);
    }
    typename iter_t::reference operator [] (size_t c) const {
      return *(row_begin + c);
    }
  private:
    iter_t row_begin;
  };

public:
  matrix() : matrix(0, 0, T()) {}
  matrix(const size_t r, const size_t c, const T& init) : rows_(r), cols_(c), data_(r * c, init) {}
  matrix(const size_t r) : matrix(r, 1, T()) {}
  matrix(const size_t r, const size_t c) : matrix(r, c, T()) {}
  matrix(const std::initializer_list<std::initializer_list<T>>& list) noexcept(false)
    : matrix(list.size(), list.begin()->size()) {
    auto iter = data_.begin();
    for (const auto& r: list) {
      if (r.size() == cols_) {
        for (const auto& i : r) {
          *iter = i;
          ++iter;
        }
      } else {
        throw std::range_error("Row lengths are different in the initialization list.");
      }
    }
  }

  matrix(const matrix& /*m*/) = default;
  matrix(matrix&& /*m*/) = default;

  matrix& operator = (const matrix& m) {
    rows_ = m.rows_;
    cols_ = m.cols_;
    data_ = m.data_;
    return *this;
  }

  matrix& operator = (matrix&& m) {
    std::swap(rows_, m.rows_);
    std::swap(cols_, m.cols_);
    data_.swap(m.data_);
    return *this;
  }

  size_t cols() const { return cols_; }  // number of columns
  size_t rows() const { return rows_; }  // number of rows

  bool empty() const noexcept { return data_.empty(); }

  matrix<T>& swap_row(size_t r1, size_t r2) {
    auto a = data_.begin() + r1 * cols_;
    auto b = data_.begin() + r2 * cols_;
    std::swap_ranges(a, a + cols_, b);
    return *this;
  }

  bool operator == (const matrix& m) const {
    if (m.rows_ == rows_ && m.cols_ == cols_) {
      return m.data_ == data_;
    }
    return false;
  }
  bool operator != (const matrix& m) const {
    return !(m == *this);
  }

  const row<const_iterator> operator [] (size_t r) const {
    return row<const_iterator>{data_.cbegin() + (r * cols_)};
  }
  row<iterator> operator [] (size_t r) {
    return row<iterator>{data_.begin() + (r * cols_)};
  }

private:
  size_t rows_;
  size_t cols_;
  data_container data_;  // Array of matrix numbers
};

template <typename T, template <typename> typename Matrix>
static inline typename Matrix<T> transpose(const typename Matrix<T>& m) {
  typename Matrix<T> rm{m.cols(), m.rows()};
  for (size_t i = 0; i < rm.rows(); i++) {
    for (size_t j = 0; j < rm.cols(); j++) {
      rm[i][j] = m[j][i];
    }
  }
  return rm;
}

template <typename T>
static inline matrix<T> operator * (const matrix<T>& m1, const matrix<T>& m2) noexcept(false) {
  if (m1.cols() == m2.rows()) {
    matrix<T> rm{m1.rows(), m2.cols()};
    for (size_t i = 0; i < m1.rows(); i++) {
      for (size_t j = 0; j < m2.cols(); j++) {
        rm[i][j] = 0;
        for (size_t k = 0; k < m1.cols(); k++) {
          rm[i][j] += m1[i][k] * m2[k][j];
        }
      }
    }
    return rm;
  }
  throw std::range_error("For matrix multiplication cols of M1 shall be equals rows of M2.");
}

template <typename T>
static inline matrix<T> operator + (const matrix<T>& m1, const matrix<T>& m2) noexcept(false) {
  if (m1.rows() == m2.rows() && m1.cols() == m2.cols()) {
    matrix<T> rm{m1.rows(), m1.cols()};
    for (size_t i = 0; i < m1.rows(); i++) {
      for (size_t j = 0; j < m1.cols(); j++) {
        rm[i][j] = m1[i][j] + m2[i][j];
      }
    }
    return rm;
  }
  throw std::range_error("For matrix addition the sizes of both matrix shall be equals.");
}

template <typename T>
static inline matrix<T> operator - (const matrix<T>& m1, const matrix<T>& m2) noexcept(false) {
  if (m1.rows() == m2.rows() && m1.cols() == m2.cols()) {
    matrix<T> rm{m1.rows(), m1.cols()};
    for (size_t i = 0; i < m1.rows(); i++) {
      for (size_t j = 0; j < m1.cols(); j++) {
        rm[i][j] = m1[i][j] - m2[i][j];
      }
    }
    return rm;
  }
  throw std::range_error("For matrix subtraction the sizes of both matrix shall be equals.");
}

template <typename T>
static inline matrix<T> operator - (const matrix<T>& m) noexcept(true) {
  matrix<T> rm{m.rows(), m.cols()};
  for (size_t i = 0; i < m.rows(); i++) {
    for (size_t j = 0; j < m.cols(); j++) {
      rm[i][j] = -m[i][j];
    }
  }
  return rm;
}

}  // namespace mathlib

#endif  // MATHLIB_MATRIX_H
