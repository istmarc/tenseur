#ifndef TENSEUR_IOS_TENSOR
#define TENSEUR_IOS_TENSOR

namespace ten {

/// Overload for scalar
template <typename T>
std::ostream &operator<<(std::ostream &os, const ::ten::scalar<T> &s) {
   os << s.value();
   return os;
}

/// Overload << operator for vector
template <class T, class Shape, storage_order order, class Storage,
          class Allocator>
   requires(::ten::is_dynamic_vector<
            ranked_tensor<T, Shape, order, Storage, Allocator>>::value)
std::ostream &
operator<<(std::ostream &os,
           const ranked_tensor<T, Shape, order, Storage, Allocator> &t) {
   os << "vector<" << ::ten::to_string<T>() << "," << t.shape() << ">";
   size_type size = t.size();
   if (size <= 10) {
      for (size_type i = 0; i < t.size(); i++) {
         os << "\n" << t[i];
      }
   } else {
      for (size_type i = 0; i < 5; i++) {
         os << "\n" << t[i];
      }
      os << "\n⋮";
      for (size_type i = t.size() - 5; i < t.size(); i++) {
         os << "\n" << t[i];
      }
   }
   return os;
}

/// Overload << operator for a static vector
template <class T, class Shape, storage_order order, class Storage,
          class Allocator>
   requires(::ten::is_svector<
            ranked_tensor<T, Shape, order, Storage, Allocator>>::value)
std::ostream &
operator<<(std::ostream &os,
           const ranked_tensor<T, Shape, order, Storage, Allocator> &t) {
   os << "svector<" << ::ten::to_string<T>() << "," << Shape::static_size()
      << ">";
   size_type size = t.size();
   if (size <= 10) {
      for (size_type i = 0; i < t.size(); i++) {
         os << "\n" << t[i];
      }
   } else {
      for (size_type i = 0; i < 5; i++) {
         os << "\n" << t[i];
      }
      os << "\n⋮";
      for (size_type i = t.size() - 5; i < t.size(); i++) {
         os << "\n" << t[i];
      }
   }
   return os;
}

/// Overload << operator for matrix
template <class T, class Shape, storage_order order, class Storage,
          class Allocator>
   requires(::ten::is_matrix<
            ranked_tensor<T, Shape, order, Storage, Allocator>>::value)
std::ostream &
operator<<(std::ostream &os,
           const ranked_tensor<T, Shape, order, Storage, Allocator> &t) {
   os << "matrix<" << ::ten::to_string<T>() << "," << t.shape() << ">";
   size_type m = t.dim(0);
   size_type n = t.dim(1);
   for (size_type i = 0; i < m; i++) {
      os << "\n";
      os << t(i, 0);
      for (size_type j = 1; j < n; j++) {
         os << "   " << t(i, j);
      }
   }
   return os;
}

/// Overload << operator for static matrix
template <class T, class Shape, storage_order order, class Storage,
          class Allocator>
   requires(::ten::is_smatrix<
            ranked_tensor<T, Shape, order, Storage, Allocator>>::value)
std::ostream &
operator<<(std::ostream &os,
           const ranked_tensor<T, Shape, order, Storage, Allocator> &t) {
   os << "smatrix<" << ::ten::to_string<T>() << ","
      << Shape::template static_dim<0>() << "x"
      << Shape::template static_dim<1>() << ">";
   size_type m = Shape::template static_dim<0>();
   size_type n = Shape::template static_dim<1>();
   for (size_type i = 0; i < m; i++) {
      os << "\n";
      os << t(i, 0);
      for (size_type j = 1; j < n; j++) {
         os << "   " << t(i, j);
      }
   }
   return os;
}

/// Overload << operator for diagonal matrix
template <class T, class Shape, storage_order order, class Storage,
          class Allocator>
   requires(::ten::is_diagonal<
            ranked_tensor<T, Shape, order, Storage, Allocator>>::value)
std::ostream &
operator<<(std::ostream &os,
           const ranked_tensor<T, Shape, order, Storage, Allocator> &t) {
   os << "diagonal<" << ::ten::to_string<T>() << "," << t.shape() << ">";
   size_type size = t.size();
   if (size <= 10) {
      for (size_type i = 0; i < size; i++) {
         os << "\n" << t[i];
      }
   } else {
      for (size_type i = 0; i < 5; i++) {
         os << "\n" << t[i];
      }
      os << "⋮\n";
      for (size_type i = size - 5; i < size; i++) {
         os << "\n" << t[i];
      }
   }

   return os;
}

/// Overload << operator for static diagonal matrix
template <class T, class Shape, storage_order order, class Storage,
          class Allocator>
   requires(::ten::is_sdiagonal<
            ranked_tensor<T, Shape, order, Storage, Allocator>>::value)
std::ostream &
operator<<(std::ostream &os,
           const ranked_tensor<T, Shape, order, Storage, Allocator> &t) {
   os << "sdiagonal<" << ::ten::to_string<T>() << ","
      << Shape::template static_dim<0>() << "x"
      << Shape::template static_dim<1>() << ">";
   size_type size = t.size();
   if (size <= 10) {
      for (size_type i = 0; i < size; i++) {
         os << "\n" << t[i];
      }
   } else {
      for (size_type i = 0; i < 5; i++) {
         os << "\n" << t[i];
      }
      os << "⋮\n";
      for (size_type i = size - 5; i < size; i++) {
         os << "\n" << t[i];
      }
   }
   return os;
}

/// Overload << operator for column
template <class T, class Shape, storage_order order, class Storage,
          class Allocator>
   requires(::ten::is_dynamic_column<
            ranked_column<T, Shape, order, Storage, Allocator>>::value)
std::ostream &
operator<<(std::ostream &os,
           const ranked_column<T, Shape, order, Storage, Allocator> &t) {
   os << "column<" << ::ten::to_string<T>() << "," << t.size() << ">";
   size_type size = t.size();
   if (size <= 10) {
      for (size_type i = 0; i < t.size(); i++) {
         os << "\n" << t[i];
      }
   } else {
      for (size_type i = 0; i < 5; i++) {
         os << "\n" << t[i];
      }
      os << "\n⋮";
      for (size_type i = t.size() - 5; i < t.size(); i++) {
         os << "\n" << t[i];
      }
   }
   return os;
}

/// Overload << operator for column
template <class T, class Shape, storage_order order, class Storage,
          class Allocator>
   requires(::ten::is_scolumn<
            ranked_column<T, Shape, order, Storage, Allocator>>::value)
std::ostream &
operator<<(std::ostream &os,
           const ranked_column<T, Shape, order, Storage, Allocator> &t) {
   os << "scolumn<" << ::ten::to_string<T>() << "," << t.size() << ">";
   size_type size = t.size();
   if (size <= 10) {
      for (size_type i = 0; i < t.size(); i++) {
         os << "\n" << t[i];
      }
   } else {
      for (size_type i = 0; i < 5; i++) {
         os << "\n" << t[i];
      }
      os << "\n⋮";
      for (size_type i = t.size() - 5; i < t.size(); i++) {
         os << "\n" << t[i];
      }
   }
   return os;
}

/// Overload << operator for row
template <class T, class Shape, storage_order order, class Storage,
          class Allocator>
   requires(::ten::is_dynamic_row<
            ranked_row<T, Shape, order, Storage, Allocator>>::value)
std::ostream &
operator<<(std::ostream &os,
           const ranked_row<T, Shape, order, Storage, Allocator> &t) {
   os << "row<" << ::ten::to_string<T>() << "," << t.size() << ">\n";
   size_type size = t.size();
   if (size <= 10) {
      os << t[0];
      for (size_type i = 1; i < t.size(); i++) {
         os << " " << t[i];
      }
   } else {
      os << t[0];
      for (size_type i = 0; i < 5; i++) {
         os << " " << t[i];
      }
      os << "\n...";
      for (size_type i = t.size() - 5; i < t.size(); i++) {
         os << " " << t[i];
      }
   }
   return os;
}

/// Overload << operator for row
template <class T, class Shape, storage_order order, class Storage,
          class Allocator>
   requires(
       ::ten::is_srow<ranked_row<T, Shape, order, Storage, Allocator>>::value)
std::ostream &
operator<<(std::ostream &os,
           const ranked_row<T, Shape, order, Storage, Allocator> &t) {
   os << "srow<" << ::ten::to_string<T>() << "," << t.size() << ">\n";
   size_type size = t.size();
   if (size <= 10) {
      os << t[0];
      for (size_type i = 1; i < size; i++) {
         os << " " << t[i];
      }
   } else {
      os << t[0];
      for (size_type i = 0; i < 5; i++) {
         os << " " << t[i];
      }
      os << "\n...";
      for (size_type i = size - 5; i < size; i++) {
         os << " " << t[i];
      }
   }
   return os;
}

} // namespace ten
#endif
