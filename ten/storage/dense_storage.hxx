#ifndef TENSEUR_STORAGE_DENSE_STORAGE
#define TENSEUR_STORAGE_DENSE_STORAGE

#include <memory>
#include <new>
#include <type_traits>

#include <ten/types.hxx>

namespace ten {
/// \class dense_storage
/// Dense array
template <typename T, typename allocator> class dense_storage final {
 public:
   using value_type = T;
   using allocator_type = allocator;
   using allocator_traits = std::allocator_traits<allocator>;

   template <class to> using casted_type = dense_storage<to, allocator>;

 private:
   allocator_type _allocator{};
   size_type _size = 0;
   T *_data = nullptr;

 public:
   dense_storage() noexcept {}

   template<typename Shape>
   dense_storage(const Shape& shape) noexcept {
       _size = shape.size();
      _data = allocator_traits::allocate(_allocator, _size * sizeof(T));
   }

   ~dense_storage() {
      if (_data)
         delete[] _data;
   }

   [[nodiscard]] inline const T *data() const { return _data; }

   [[nodiscard]] inline T *data() { return _data; }

   [[nodiscard]] inline size_type size() const {return _size;}

   /// Get/Set the element at index
   [[nodiscard]] inline const T &operator[](size_type index) const noexcept {
      return _data[index];
   }
   [[nodiscard]] inline T &operator[](size_type index) noexcept {
      return _data[index];
   }
};

/// \class sdense_storage
/// Static dense array
template <typename T, size_t N> class sdense_storage final {
 public:
   using value_type = T;

   template <class to> using casted_type = sdense_storage<to, N>;

   using allocator_type = void;

 private:
   alignas(T) T _data[N];

 public:
   sdense_storage() noexcept {}

   ~sdense_storage() {}

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

   [[nodiscard]] inline static constexpr size_type size() {return N;}

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
