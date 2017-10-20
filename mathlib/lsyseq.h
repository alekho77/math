/*
    Solving system of linear equations.
    (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_LSYSEQ_H
#define MATHLIB_LSYSEQ_H

#include "matrix.h"

namespace mathlib {

// [A]*[X]=[B]
template <typename T>
class linear_equations {
  static_assert(std::is_floating_point<T>::value == true, "Solving system of linear equations is possible only for floating point numbers.");
public:
  linear_equations() = delete;
  linear_equations(matrix<T>&& a, matrix<T>&& b) noexcept(false) : X_(a.rows()), A_(a), B_(b) {
    if (A_.rows() < A_.cols()) {
      throw std::invalid_argument("Number of equations is less than variables. The system has no solution.");
    } else if (A_.rows() > A_.cols()) {
      throw std::invalid_argument("Number of equations is greater than variables. The system has infinitely many solutions.");
    }
    if (B_.cols() != 1 || B_.rows() != A_.rows()) {
      throw std::invalid_argument("Matrix of the constant terms shall have only one column.");
    }
  }
  linear_equations(const matrix<T>& a, const matrix<T>& b)
    : linear_equations(matrix<T>(a), matrix<T>(b)) {
  }
  ~linear_equations() = default;

  const matrix<T>& A() const { return A_; }
  const matrix<T>& B() const { return B_; }
  const matrix<T>& X() const { return X_; }

  // Make matrix of the coefficients top-triangular.
  linear_equations<T>& normalize() noexcept(false);

  // Calculate the system conditionality. The system should be normalized previously.
  T cond() const;

  // Solve the system. The system should be normalized previously.

private:
  matrix<T> X_;
  matrix<T> A_;
  matrix<T> B_;
};

template <typename T>
linear_equations<T>& linear_equations<T>::normalize() noexcept(false) {
  for (size_t i = 0; i < A_.rows(); i++) {
    // Find max coefficient in i-column
    size_t r_max = i;
    for (size_t r = i + 1; r < A_.rows(); r++) {
      if (std::abs(A_[r_max][i]) < std::abs(A_[r][i])) {
        r_max = r;
      }
    }
    if (r_max != i) {
      A_.swap_row(i, r_max);
      B_.swap_row(i, r_max);
    }
    if (A_[i][i] == 0) {
      throw std::underflow_error("Main coefficient is null.");
    }
    // Turning coefficients below i-row
    for (size_t r = i + 1; r < A_.rows(); r++) {
      if (A_[r][i] == 0) {
        continue;
      }
      const T fact = -A_[r][i] / A_[i][i];
      for (size_t k = i + 1; k < A_.cols(); k++) {
        A_[r][k] += A_[i][k] * fact;
      }
      B_[r][0] += B_[i][0] * fact;
      A_[r][i] = 0;
    }
  }
  return *this;
}

template <typename T>
T linear_equations<T>::cond() const {
  T mu_min = std::abs(A_[0][0]);
  T mu_max = std::abs(A_[0][0]);
  for (size_t i = 1; i < A_.rows(); i++) {
    mu_min = std::min(mu_min, std::abs(A_[i][i]));
    mu_max = std::max(mu_max, std::abs(A_[i][i]));
  }
  return mu_max / mu_min;  // Since triag-matrix has been solved before, so all coeff in diag are greater than null.
}

}  // namespace mathlib

/*  Вычисляет детерминант от верхней треугольной матрицы. */
//Extended __fastcall Det_qs(Matrix& A);
/*  Решение системы.
    Возврашает матрицу-столбец с решением. */
//Matrix __fastcall DeqSys(const Matrix& A, const Matrix& B, int& iErr);
/*  Вычисление невязки решения системы линейных уравнений
    Возвращает матрицу-столбец с невязкой по каждому из корней */
//Matrix __fastcall Inex(const Matrix& A, const Matrix& X, const Matrix& B);
/*  Вычисление среднеквадратичной невязки решения системы линейных уравнений
    Возвращает число с невязкой */
//Extended __fastcall SqrtInex(const Matrix& A, const Matrix& X, const Matrix& B);
/*  Транспонирование матрицы
    Возвращает транспонированную матрицу */
//Matrix __fastcall Trans(Matrix& M);
//---------------------------------------------------------------------------
/* Получить строку с описанием ошибки ErrCode */
//AnsiString __fastcall DeqSysErrStr(int ErrCode);
//---------------------------------------------------------------------------
#endif  // MATHLIB_LSYSEQ_H
 