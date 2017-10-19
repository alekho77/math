/*
    Object "Matrix" for linear algebra.
    (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_MATRIX_H
#define MATHLIB_MATRIX_H

#include <vector>
#include <algorithm>

namespace mathlib {

template <typename T>
class matrix
{
  class crow {
  public:
    crow() = delete;
    crow(crow&& /*r*/) = default;
    crow(const crow& /*r*/) = default;
    crow(const T* data) : data_(data) {}
    const T& operator [] (size_t c) const {
      return data_[c];
    }
  private:
    const T* data_;
  };

  class row {
  public:
    row() = delete;
    row(row&& /*r*/) = default;
    row(const row& /*r*/) = default;
    row(T* data) : data_(data) {}
    T& operator [] (size_t c) {
      return data_[c];
    }
  private:
    T* data_;
  };

public:
  matrix() = delete;
  matrix(const size_t r, const size_t c, const T& init)
    : rows_(r)
    , cols_(c)
    , data_(r * c, init) {
  }
  matrix(const size_t r) : matrix(r, 1, T()) {}
  matrix(const size_t r, const size_t c) : matrix(r, c, T()) {}
  matrix(const std::initializer_list<std::initializer_list<T>>& list) noexcept(false)
    : rows_(list.size())
    , cols_(list.begin()->size())
    , data_(list.size() * list.begin()->size()) {
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
  ~matrix() = default;

  size_t cols() const { return cols_; }  // number of columns
  size_t rows() const { return rows_; }  // number of rows

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

  crow operator [] (size_t r) const {
    return crow{&data_[r * cols_]};
  }
  row operator [] (size_t r) {
    return row{&data_[r * cols_]};
  }

private:
  const size_t rows_;
  const size_t cols_;
  std::vector<typename T> data_;  // Array of matrix numbers
};

template <typename T>
static inline matrix<T> transpose(const matrix<T>& m) {
  matrix<T> rm{m.cols(), m.rows()};
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

}  // namespace mathlib

#endif  // MATHLIB_MATRIX_H
