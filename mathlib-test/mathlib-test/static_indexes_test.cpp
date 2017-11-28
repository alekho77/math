#include "math/mathlib/static_indexes.h"

namespace mathlib {

namespace {
using indexes1_t = index_pack<1, 2, 3, 4, 5>;
using indexes2_t = index_pack<5, 4, 3>;
using indexes3_t = index_pack<3, 2>;
using pack_t = type_pack<indexes1_t, indexes2_t, indexes3_t>;

constexpr size_t i1size = indexes1_t::size;
constexpr size_t i2size = indexes2_t::size;
constexpr size_t i3size = indexes3_t::size;
static_assert(5 == i1size, "Invalid index array size");
static_assert(3 == i2size, "Invalid index array size");
static_assert(2 == i3size, "Invalid index array size");

constexpr size_t psize = pack_size<pack_t>();
static_assert(3 == psize, "Invalid type pack size");

constexpr size_t cidx1 = get_index<indexes1_t, 3>();
constexpr size_t cidx2 = get_index<indexes2_t, 2>();
constexpr size_t cidx3 = get_index<indexes3_t, 1>();
static_assert(4 == cidx1, "Wrong index");
static_assert(3 == cidx2, "Wrong index");
static_assert(2 == cidx3, "Wrong index");

constexpr size_t cidx4 = get_index<get_type_t<pack_t, 1>, 1>();
static_assert(4 == cidx4, "Wrong index");
}

}  // namespace mathlib
