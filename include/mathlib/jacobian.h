/*
    Simple way to work with the Jacobian matrix.
    (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_JACOBIAN_H
#define MATHLIB_JACOBIAN_H

#include "derivative.h"
#include "fmatrix.h"

namespace mathlib {

template <typename F> class jacobian;

template <typename R, typename... Args> class jacobian<R(Args...)> {
    using function_t = std::function<R(Args...)>;
    using derivative_t = derivative<R(Args...)>;

 public:
    explicit jacobian(size_t rows) : W_(rows, sizeof...(Args)), derivatives_(rows) {}
    jacobian(const std::initializer_list<function_t>& flist) : jacobian(flist.size()) {
        auto fiter = flist.begin();
        for (size_t i = 0; i < flist.size(); i++, ++fiter) {
            initialize_row(i, *fiter);
        }
    }
    jacobian& initialize_row(size_t idx, const function_t& fun) {
        make_derivatives(fun, idx, std::index_sequence_for<Args...>{});
        return *this;
    }
    matrix<R> operator()(Args... args) const {
        return W_(args...);
    }

 private:
    template <size_t... I> void make_derivatives(const function_t& fun, size_t idx, std::index_sequence<I...>) {
        derivatives_[idx] = derivative_t{fun};
        make_jacobian_row(idx, {make_derivative<I>(derivatives_[idx])...});
    }
    template <size_t K> function_t make_derivative(const derivative_t& deriv) {
        return [&deriv](Args... args) -> R { return deriv.template diff<K>(args...); };
    }
    void make_jacobian_row(size_t idx, std::initializer_list<function_t>&& list) {
        auto fiter = list.begin();
        for (size_t j = 0; j < list.size(); j++, ++fiter) {
            W_[idx][j] = *fiter;
        }
    }
    fmatrix<R(Args...)> W_;
    std::vector<derivative_t> derivatives_;
};

} // namespace mathlib

#endif // MATHLIB_JACOBIAN_H
