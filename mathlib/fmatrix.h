/*
    "Matrix" for functional objects.
    (c) 2017 Aleksey Khozin.
*/

#ifndef MATHLIB_FMATRIX_H
#define MATHLIB_FMATRIX_H

#include "matrix.h"
#include "helpers.h"

#include <functional>

namespace mathlib {

template <typename F>
class fmatrix;

template <typename R, typename... Args>
class fmatrix<R(Args...)> {
    static_assert(are_same<R, Args...>::value == true, "Different types are not allowed.");

    using function_t = std::function<R(Args...)>;
    using data_container = std::vector<typename function_t>;
    using const_iterator = typename data_container::const_iterator;
    using iterator = typename data_container::iterator;

    template <typename iter_t>
    class row {
     public:
        row() = delete;
        row(iter_t&& r) : row_begin(std::move(r)) {}
        typename iter_t::reference operator[](size_t c) {
            return *(row_begin + c);
        }
        typename iter_t::reference operator[](size_t c) const {
            return *(row_begin + c);
        }

     private:
        iter_t row_begin;
    };

 public:
    fmatrix() : fmatrix(0, 0) {}
    explicit fmatrix(const size_t r) : fmatrix(r, 1) {}
    fmatrix(const size_t r, const size_t c) : rows_(r), cols_(c), data_(r * c, typename function_t()) {}
    fmatrix(std::initializer_list<std::initializer_list<function_t>>&& list) noexcept(false)
        : fmatrix(list.size(), list.begin()->size()) {
        auto iter = data_.begin();
        for (const auto& r : list) {
            if (r.size() == cols_) {
                for (auto&& i : r) {
                    *iter = std::move(i);
                    ++iter;
                }
            } else {
                throw std::range_error("Row lengths are different in the initialization list.");
            }
        }
    }

    matrix<R> operator()(Args... args) const {
        matrix<R> res{rows_, cols_};
        for (size_t r = 0; r < rows_; r++) {
            for (size_t c = 0; c < cols_; c++) {
                res[r][c] = (*this)[r][c](args...);
            }
        }
        return res;
    }

    size_t cols() const {
        return cols_;
    }  // number of columns
    size_t rows() const {
        return rows_;
    }  // number of rows

    bool empty() const noexcept {
        return data_.empty();
    }

    const row<const_iterator> operator[](size_t r) const {
        return row<const_iterator>{data_.cbegin() + (r * cols_)};
    }
    row<iterator> operator[](size_t r) {
        return row<iterator>{data_.begin() + (r * cols_)};
    }

    fmatrix<R(Args...)>& swap_row(size_t r1, size_t r2) {
        auto a = data_.begin() + r1 * cols_;
        auto b = data_.begin() + r2 * cols_;
        std::swap_ranges(a, a + cols_, b);
        return *this;
    }

 private:
    size_t rows_;
    size_t cols_;
    data_container data_;  // Array of matrix numbers
};

template <typename R, typename... Args>
static inline fmatrix<R(Args...)> operator*(const fmatrix<R(Args...)>& m1,
                                            const fmatrix<R(Args...)>& m2) noexcept(false) {
    if (m1.cols() == m2.rows()) {
        fmatrix<R(Args...)> rm{m1.rows(), m2.cols()};
        for (size_t i = 0; i < m1.rows(); i++) {
            for (size_t j = 0; j < m2.cols(); j++) {
                rm[i][j] = [i, j, m1, m2](Args... args) -> R {
                    R res = 0;
                    for (size_t k = 0; k < m1.cols(); k++) {
                        res += m1[i][k](args...) * m2[k][j](args...);
                    }
                    return res;
                };
            }
        }
        return rm;
    }
    throw std::range_error("For matrix multiplication cols of M1 shall be equals rows of M2.");
}

template <typename R, typename... Args>
static inline fmatrix<R(Args...)> operator+(const fmatrix<R(Args...)>& m1,
                                            const fmatrix<R(Args...)>& m2) noexcept(false) {
    if (m1.rows() == m2.rows() && m1.cols() == m2.cols()) {
        fmatrix<R(Args...)> rm{m1.rows(), m1.cols()};
        for (size_t i = 0; i < m1.rows(); i++) {
            for (size_t j = 0; j < m1.cols(); j++) {
                rm[i][j] = [i, j, m1, m2](Args... args) -> R { return m1[i][j](args...) + m2[i][j](args...); };
            }
        }
        return rm;
    }
    throw std::range_error("For matrix addition the sizes of both matrix shall be equals.");
}

template <typename R, typename... Args>
static inline fmatrix<R(Args...)> operator-(const fmatrix<R(Args...)>& m1,
                                            const fmatrix<R(Args...)>& m2) noexcept(false) {
    if (m1.rows() == m2.rows() && m1.cols() == m2.cols()) {
        fmatrix<R(Args...)> rm{m1.rows(), m1.cols()};
        for (size_t i = 0; i < m1.rows(); i++) {
            for (size_t j = 0; j < m1.cols(); j++) {
                rm[i][j] = [i, j, m1, m2](Args... args) -> R { return m1[i][j](args...) - m2[i][j](args...); };
            }
        }
        return rm;
    }
    throw std::range_error("For matrix subtraction the sizes of both matrix shall be equals.");
}

}  // namespace mathlib

#endif  // MATHLIB_FMATRIX_H
