/*
  Static indexes package.
  (c) 2017 Aleksey Khozin.
*/

#ifndef MATHLIB_STATIC_INDEXES_H
#define MATHLIB_STATIC_INDEXES_H

#include <array>
#include <type_traits>

namespace mathlib {

template <size_t... I>
struct index_pack {
  static constexpr size_t size = sizeof...(I);
  static const std::array<size_t, sizeof...(I)> indexes;
};

template <size_t... I>
const std::array<size_t, sizeof...(I)> index_pack<I...>::indexes = {I...};

template <typename Pack, size_t I>
static constexpr size_t get_index() {
  return std::get<I>(Pack::indexes);
}

template <typename... P>
struct type_pack;

template <>
struct type_pack<> {};

template <typename P, typename... Rest>
struct type_pack<P, Rest...> {
  using type = P;
  using next = type_pack<Rest...>;
};

template <typename Pack, size_t I>
struct get_type {
  static_assert(!std::is_same<Pack, type_pack<>>::value, "index out of bounds");
  using type = typename get_type<typename Pack::next, I - 1>::type;
};

template <typename Pack>
struct get_type<Pack, 0> {
  static_assert(!std::is_same<Pack, type_pack<>>::value, "index out of bounds");
  using type = typename Pack::type;
};

template <typename Pack>
static constexpr size_t pack_size() {
  return 1 + pack_size<typename Pack::next>();
}

template <>
constexpr size_t pack_size<type_pack<>>() {
  return 0;
}

}  // namespace mathlib

#endif  // MATHLIB_STATIC_INDEXES_H
