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
  static constexpr size_t indexes[size] = {I...};
};

template <typename Pack, size_t I>
static constexpr size_t get_index() {
  static_assert(I < Pack::size, "Index out of bounds.");
  return Pack::indexes[I];
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
  static_assert(!std::is_same<Pack, type_pack<>>::value, "Index out of bounds.");
  using type = typename get_type<typename Pack::next, I - 1>::type;
};

template <typename Pack>
struct get_type<Pack, 0> {
  static_assert(!std::is_same<Pack, type_pack<>>::value, "Index out of bounds.");
  using type = typename Pack::type;
};

template <typename Pack, size_t I>
using get_type_t = typename get_type<Pack, I>::type;

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
