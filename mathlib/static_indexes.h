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

template <size_t I, typename Pack>
static constexpr size_t get_index() {
  static_assert(I < Pack::size, "Index out of bounds.");
  return Pack::indexes[I];
}

template <size_t N>
class make_index_pack {
  template <size_t... I>
  static auto helper(std::index_sequence<I...>) -> decltype(index_pack<I...>()) {}
public:
  using type = decltype(helper(std::make_index_sequence<N>()));
};

template <size_t N>
using index_sequence_pack_t = typename make_index_pack<N>::type;

template <typename... P>
struct type_pack;

template <>
struct type_pack<> {};

template <typename P, typename... Rest>
struct type_pack<P, Rest...> {
  using type = P;
  using next = type_pack<Rest...>;
};

template <typename P, size_t N>
class make_type_pack {
  static_assert(N > 0, "Number of repeats shall be greater zero.");
  
  template <size_t I> using next_type = P;
  template <size_t... I>
  static auto helper(std::index_sequence<I...>) -> decltype(type_pack<next_type<I>...>()) {}

public:
  using type = decltype(helper(std::make_index_sequence<N>()));
};

template <size_t I, typename Pack>
struct get_type {
  static_assert(!std::is_same<Pack, type_pack<>>::value, "Index out of bounds.");
  using type = typename get_type<I - 1, typename Pack::next>::type;
};

template <typename Pack>
struct get_type<0, Pack> {
  static_assert(!std::is_same<Pack, type_pack<>>::value, "Index out of bounds.");
  using type = typename Pack::type;
};

template <size_t I, typename Pack>
using get_type_t = typename get_type<I, Pack>::type;

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
