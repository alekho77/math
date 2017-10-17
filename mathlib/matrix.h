/*
    Object "Matrix" for linear algebra.
    (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_MATRIX_H
#define MATHLIB_MATRIX_H

#include <vector>

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

public:
  matrix() = delete;
  matrix(const size_t r, const size_t c, const T& init)
  : rows_(r)
  , cols_(c)
  , data_(r * c, init) {
  }
  matrix(const size_t r) : matrix(r, 1, T()) {}
  matrix(const size_t r, const size_t c) : matrix(r, c, T()) {}

  matrix(const matrix& /*m*/) = default;
  matrix(matrix&& /*m*/) = default;

  size_t cols() const { return cols_; }  // number of columns
  size_t rows() const { return rows_; }  // number of rows

  //Matrix& __fastcall operator = (const Matrix& MSrc);
  /* Инициализация матрицы массивом чисел (если Src==NULL высвобождение занимаемой памяти, и Lin=Col=0) */
  //Matrix& __fastcall operator = (const Extended *Src); // Необходимо зарание задать размеры
  //Matrix& __fastcall operator = (const Extended Initial); // Необходимо зарание задать размеры

  //Matrix& __fastcall operator += (const Matrix& MSum);
  //Matrix __fastcall operator + (const Matrix& MS) const;
  //Matrix& __fastcall operator -= (const Matrix& MSum);
  //Matrix __fastcall operator - (const Matrix& MS) const;

  //Matrix& __fastcall operator *= (const Extended& Val);
  //Matrix& __fastcall operator /= (const Extended& Val);
  //friend Matrix __fastcall operator * (const Extended& Val, const Matrix& MMul);
  //Matrix __fastcall operator * (const Extended& Val) const;
  //Matrix __fastcall operator / (const Extended& Val) const;

  /* Размерности матриц : M1(l,m)*M2(m,n)=M3(l,n) */
  //Matrix& __fastcall operator *= (const Matrix& MMul);
  //Matrix __fastcall operator * (const Matrix& MMul) const;

  //Matrix __fastcall operator - (void) const;

  /* Равенство только, если : Резмеры совпадают и равны между собой соответствующие элементы */
  //bool __fastcall operator == (const Matrix& MCom) const;
  //bool __fastcall operator != (const Matrix& MCom) const;
  /* Равенство только, если : Все элементы равны заданному числу */
  //bool __fastcall operator == (const Extended& Val) const;
  //bool __fastcall operator != (const Extended& Val) const;

  crow operator [] (size_t r) const {
    return crow{&data_[r * cols_]};
  }

private:
  const size_t rows_;
  const size_t cols_;
  std::vector<typename T> data_;  // Array of matrix numbers
};

}  // namespace mathlib

#endif  // MATHLIB_MATRIX_H
