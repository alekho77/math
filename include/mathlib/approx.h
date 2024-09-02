/*
  Linear approximation.
  (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_APPROX_H
#define MATHLIB_APPROX_H

#include "mathlib/helpers.h"
#include "mathlib/lsyseq.h"
#include "mathlib/matrix.h"

#include <memory>
#include <vector>

namespace mathlib {

template <typename T, size_t N> class approx {
    static_assert(N > 0, "Must be at least one variable.");

    using pack_t = typename make_tuple_type<T, N + 1>::type;
    using coef_t = typename make_tuple_type<T, N>::type;

 public:
    // Add new approach, the last arg is the const term (b).
    template <typename... Args> approx& operator()(Args&&... args) {
        static_assert(sizeof...(Args) == (N + 1), "Number of arguments must be one more than variables.");
        approaches_.push_back({std::forward<Args>(args)...});
        return *this;
    }

    approx& approach() {
        matrix<T> A{approaches_.size(), N};
        matrix<T> B{approaches_.size()};
        for (size_t i = 0; i < approaches_.size(); i++) {
            matrix_row(i, A, B, approaches_[i], std::make_index_sequence<N>());
        }
        matrix<T> AT = transpose(A);
        linear_equations<T> syseq{AT * A, AT * B};
        coefs_ = syseq.normalize().solve();
        return *this;
    }

    const matrix<T>& get_as_matrix() const {
        return coefs_;
    }

    coef_t get_as_tuple() const {
        return make_coef_tuple(std::make_index_sequence<N>());
    }

 private:
    template <size_t... I>
    void matrix_row(size_t idx, matrix<T>& A, matrix<T>& B, const pack_t& pack, std::index_sequence<I...>) {
        A[idx] = {std::get<I>(pack)...};
        B[idx][0] = std::get<N>(pack);
    }

    template <size_t... I> coef_t make_coef_tuple(std::index_sequence<I...>) const {
        return std::make_tuple(coefs_[I][0]...);
    }

    std::vector<pack_t> approaches_;
    matrix<T> coefs_;
};

} // namespace mathlib

#endif // MATHLIB_APPROX_H
