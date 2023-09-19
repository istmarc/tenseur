#ifndef TENSEUR_STORAGE_DENSE_HXX
#define TENSEUR_STORAGE_DENSE_HXX

#include <memory>
#include <new>
#include <type_traits>

#include <Ten/Types.hxx>

namespace ten {
/// \class DenseStorage
/// Dense array
template <typename T, typename Allocator> class DenseStorage final {
 public:
   using value_type = T;
   using allocator_type = Allocator;
   using allocator_traits = std::allocator_traits<Allocator>;

   template <class To> using casted_type = DenseStorage<To, Allocator>;

 private:
   allocator_type _allocator{};
   size_type _size = 0;
   T *_data = nullptr;

 public:
   DenseStorage() noexcept {}

   DenseStorage(size_type size) noexcept
       : _size(size),
         _data(allocator_traits::allocate(_allocator, size * sizeof(T))) {}

   ~DenseStorage() {
      if (_data)
         delete[] _data;
   }

   [[nodiscard]] inline const T *data() const { return _data; }

   [[nodiscard]] inline T *data() { return _data; }

   /// Get/Set the element at index
   [[nodiscard]] inline const T &operator[](size_type index) const noexcept {
      return _data[index];
   }
   [[nodiscard]] inline T &operator[](size_type index) noexcept {
      return _data[index];
   }
};

/// \class StaticDenseStorage
/// Static dense array
template <typename T, size_t N> class StaticDenseStorage final {
 public:
   using value_type = T;

   template <class To> using casted_type = StaticDenseStorage<To, N>;

   using allocator_type = void;

 private:
   alignas(T) T _data[N];

 public:
   StaticDenseStorage() noexcept {}

   ~StaticDenseStorage() {}

   // FIXME reinterpret cast aligned memory T[] to a pointer T*
   //[[nodiscard]] inline (const Scalar)[N] data() const {
   //  return std::launder(reinterpret_cast<const Scalar(*)[N]>(&_data));
   //}
   [[nodiscard]] inline const T *data() const {
      return std::launder(reinterpret_cast<const T *>(&_data));
   }

   [[nodiscard]] inline T *data() {
      return std::launder(reinterpret_cast<T *>(&_data));
   }

   /// Get/Set the element at index
   [[nodiscard]] inline const T &operator[](size_t index) const noexcept {
      return *std::launder(reinterpret_cast<const T *>(&_data[index]));
   }
   [[nodiscard]] inline T &operator[](size_t index) noexcept {
      return *std::launder(reinterpret_cast<T *>(&_data[index]));
   }
};

} // namespace ten

#endif
