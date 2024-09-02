/*
  Non-linear approximation.
  (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_FAPPROX_H
#define MATHLIB_FAPPROX_H

#include "helpers.h"
#include "jacobian.h"
#include "lsyseq.h"

namespace mathlib {

template <typename F> class fapprox;

template <typename R, typename... Args> class fapprox<R(Args...)> {
    static_assert(sizeof...(Args) > 0, "Must be at least one variable.");
    static_assert(are_same<R, Args...>::value == true, "All types must be the same.");

    using function_t = std::function<R(Args...)>;

 public:
    // Add new approach, the last arg is the const term (b).
    fapprox& operator()(const function_t& fun) {
        approaches_.push_back(fun);
        return *this;
    }

    fapprox& approach(Args... args) {
        return approach(transpose(matrix<R>{{args...}}), numeric_consts<R>::epsilon);
    }

    fapprox& approach(matrix<R> x, const R epsilon) {
        jacobian<R(Args...)> W{approaches_.size()};
        fmatrix<R(Args...)> F{approaches_.size()};
        for (size_t i = 0; i < approaches_.size(); i++) {
            F[i][0] = approaches_[i];
            W.initialize_row(i, approaches_[i]);
        }
        x += make_syseq(W, F, x, std::index_sequence_for<Args...>{}).normalize().solve();
        R eps = residual(F, 1, x);
        while (eps > epsilon) {
            matrix<R> xt = x + make_syseq(W, F, x, std::index_sequence_for<Args...>{}).normalize().solve();
            R epst = residual(F, 1, xt);
            if (epst >= eps) {
                break;
            }
            x = xt;
            eps = epst;
        }
        coefs_ = x;
        return *this;
    }

    const matrix<R>& get_as_matrix() const {
        return coefs_;
    }

    std::tuple<Args...> get_as_tuple() const {
        return make_coef_tuple(std::index_sequence_for<Args...>());
    }

 private:
    R residual(const fmatrix<R(Args...)>& F, int n, Args... args) const {
        matrix<R> e = F(args...);
        R r = 0;
        for (size_t i = 0; i < e.rows(); i++) {
            r += powi(std::abs(e[i][0]), n);
        }
        return std::pow(r, static_cast<R>(1) / static_cast<R>(n));
    }

    R residual(const fmatrix<R(Args...)>& F, int n, const matrix<R>& x) const {
        if (x.rows() != sizeof...(Args)) {
            throw std::invalid_argument("Number of results shall be equal number of variables.");
        }
        return residual_adapter(F, n, x, std::index_sequence_for<Args...>{});
    }

    template <size_t... I>
    R residual_adapter(const fmatrix<R(Args...)>& F, int n, const matrix<R>& x, std::index_sequence<I...>) const {
        return residual(F, n, x[I][0]...);
    }

    template <size_t... I>
    linear_equations<R> make_syseq(const jacobian<R(Args...)>& W, const fmatrix<R(Args...)>& F, const matrix<R>& x,
                                   std::index_sequence<I...>) const {
        using namespace std;
        auto w = W(x[I][0]...);
        auto wt = transpose(w);
        w = wt * w;
        auto f = -F(x[I][0]...);
        f = wt * f;
        for (size_t i = 0; i < w.rows(); i++) {
            const R min_d =
                (max)(numeric_consts<R>::step,
                      abs(f[i][0] / (numeric_consts<R>::increment * max(numeric_consts<R>::increment, abs(x[i][0])))));
            if (abs(w[i][i]) < min_d) {
                w[i][i] = copysign(min_d, w[i][i]);
            }
        }
        return linear_equations<R>(w, f);
    }

    template <size_t... I> std::tuple<Args...> make_coef_tuple(std::index_sequence<I...>) const {
        return std::make_tuple(coefs_[I][0]...);
    }

    std::vector<function_t> approaches_;
    matrix<R> coefs_;
};

} // namespace mathlib

#endif // MATHLIB_FAPPROX_H
