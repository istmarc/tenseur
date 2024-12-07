#ifndef TENSEUR_STORAGE_DIAGONAL_STORAGE
#define TENSEUR_STORAGE_DIAGONAL_STORAGE

#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>

#include <ten/types.hxx>

namespace ten {
/// \class DiagonalStorage
/// Diagonal array
template <typename T, typename Allocator> class diagonal_storage final {
 public:
   using value_type = T;
   using allocator_type = Allocator;
   using allocator_traits = std::allocator_traits<Allocator>;

   template <class to> using casted_type = diagonal_storage<to, Allocator>;

 private:
   allocator_type _allocator{};
   size_type _size = 0;
   T *_data = nullptr;

 public:
   diagonal_storage() noexcept {}

   template <class Shape>
   // TODO Shape is 2dim
   diagonal_storage(const Shape &shape) noexcept {
      if (shape.dim(0) != shape.dim(1)) {
         std::cerr << "Warning: Diagonal matrix must be square\n";
      }
      size_t m = shape.dim(0);
      _size = m;
      _data = allocator_traits::allocate(_allocator, _size * sizeof(T));
   }

   ~diagonal_storage() {
      if (_data)
         delete[] _data;
   }

   [[nodiscard]] inline const T *data() const { return _data; }

   [[nodiscard]] inline T *data() { return _data; }

   [[nodiscard]] inline size_type size() const { return _size; }

   /// Get/Set the element at index
   [[nodiscard]] inline const T &operator[](size_type index) const noexcept {
      return _data[index];
   }
   /// Get/Set the element at index
   [[nodiscard]] inline T &operator[](size_type index) noexcept {
      return _data[index];
   }
};

/// \class sdiagonal_storage
/// Static diagonal array
template <typename T, size_t N> class sdiagonal_storage final {
 public:
   using value_type = T;

   template <class to> using casted_type = sdiagonal_storage<to, N>;

   using allocator_type = void;

 private:
   alignas(T) T _data[N];

 public:
   sdiagonal_storage() noexcept {}

   ~sdiagonal_storage() {}

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
