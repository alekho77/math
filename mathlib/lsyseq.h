/*
    Solving system of linear equations.
    (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_LSYSEQ_H
#define MATHLIB_LSYSEQ_H

#include "helpers.h"
#include "matrix.h"

#include <cmath>

namespace mathlib {

// [A]*[X]=[B]
template <typename T> class linear_equations {
    static_assert(std::is_floating_point<T>::value == true,
                  "Solving system of linear equations is possible only for floating point numbers.");

 public:
    linear_equations() = delete;
    linear_equations(matrix<T>&& a, matrix<T>&& b) noexcept(false) : X_(a.rows()), A_(a), B_(b) {
        if (A_.rows() < A_.cols()) {
            throw std::invalid_argument("Number of equations is less than variables. The system has no solution.");
        } else if (A_.rows() > A_.cols()) {
            throw std::invalid_argument(
                "Number of equations is greater than variables. The system has infinitely many solutions.");
        }
        if (B_.cols() != 1 || B_.rows() != A_.rows()) {
            throw std::invalid_argument("Matrix of the constant terms shall have only one column.");
        }
    }
    linear_equations(const matrix<T>& a, const matrix<T>& b) : linear_equations(matrix<T>(a), matrix<T>(b)) {}

    const matrix<T>& A() const {
        return A_;
    }
    const matrix<T>& B() const {
        return B_;
    }
    const matrix<T>& X() const {
        return X_;
    }

    // Make matrix of the coefficients top-triangular.
    linear_equations<T>& normalize() noexcept(false);

    // Calculate the system conditionality (condition number). The system should be normalized previously.
    T cond() const;

    // Solve the system. The system should be normalized previously.
    const matrix<T>& solve() noexcept(false);

    // Calculate the residual of the system solution.
    T residual(int n = 1) const;

 private:
    matrix<T> X_;
    matrix<T> A_;
    matrix<T> B_;
};

template <typename T> linear_equations<T>& linear_equations<T>::normalize() noexcept(false) {
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

template <typename T> T linear_equations<T>::cond() const {
    T mu_min = std::abs(A_[0][0]);
    T mu_max = std::abs(A_[0][0]);
    for (size_t i = 1; i < A_.rows(); i++) {
        mu_min = std::min(mu_min, std::abs(A_[i][i]));
        mu_max = std::max(mu_max, std::abs(A_[i][i]));
    }
    return mu_max / mu_min; // Since triag-matrix has been solved before, so all coeff in diag are greater than null.
}

template <typename T> const matrix<T>& linear_equations<T>::solve() noexcept(false) {
    X_ = B_;
    for (int i = static_cast<int>(A_.rows()) - 1; i >= 0; i--) {
        for (int j = static_cast<int>(A_.cols()) - 1; j > i; j--) {
            X_[i][0] -= X_[j][0] * A_[i][j];
        }
        X_[i][0] /= A_[i][i];
    }
    return X_;
}

template <typename T> T linear_equations<T>::residual(int n /*= 1*/) const {
    matrix<T> R = B_ - A_ * X_;
    T res = 0;
    for (size_t i = 0; i < R.rows(); i++) {
        res += powi(std::abs(R[i][0]), n);
    }
    return std::pow(res, static_cast<T>(1) / static_cast<T>(n));
}

} // namespace mathlib

#endif // MATHLIB_LSYSEQ_H
