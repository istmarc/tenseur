#include <iostream>
#include <memory>
#include <type_traits>

#include <ten/tensor>

template <class T> void print_tensor(const T &val) {
   std::cout << "[";
   for (size_t i = 0; i < val.size(); i++)
      std::cout << val[i] << " ";
   std::cout << "]\n";
}

void print_line() {
   for (size_t i = 0; i < 18; i++)
      std::cout << "-";
   std::cout << "\n";
}

int main() {
   using namespace ten;
   using namespace std;

   {
      print_line();
      shape<dynamic, 3, 4> s({2, 3, 4});

      std::cout << "Number of dims\n";
      std::cout << s.rank() << std::endl;

      std::cout << "Dynamic shape at 0\n";
      std::cout << s.is_dynamic_dim<0>() << std::boolalpha << std::endl;
      cout << "static value = " << s.static_dim<0>() << endl;
      cout << "dynamic value = " << s.dim(0) << endl;

      cout << "Static shape at 1\n";
      cout << s.is_dynamic_dim<1>() << std::boolalpha << endl;
      cout << s.dim(1) << endl;

      cout << "Static shape at 2\n";
      cout << s.is_dynamic_dim<2>() << std::boolalpha << endl;
      cout << s.dim(2) << endl;
   }

   {
      print_line();
      cout << "Shape of static dim 2 x 3" << endl;
      shape<2, 3> a;
      cout << a.static_dim<0>() << endl;
      cout << a.static_dim<1>() << endl;
      cout << "Is dynamic = " << std::boolalpha << a.is_dynamic() << endl;
      cout << a.dim(0) << endl;
      cout << a.dim(1) << endl;
   }

   {
      print_line();
      cout << "\nDyanmic or static shape\n";
      cout << "s\n";
      shape<dynamic, 2> s({2, 2});
      cout << s.is_dynamic() << endl;
      cout << s.is_static() << endl;

      cout << "a\n";
      shape<3, 3, 5, 2, dynamic> a({3, 3, 5, 2, 4});
      cout << a.is_dynamic() << endl;
      cout << a.is_static() << endl;

      cout << "b\n";
      shape<3, 3, 5, 2> b;
      cout << b.is_dynamic() << endl;
      cout << b.is_static() << endl;
   }

   {
      print_line();
      cout << "\nStorage\n";
      dynamic_shape<2> shape({2, 3});
      dense_storage<float, std::allocator<float>> s(dynamic_shape<1>({shape.size()}));
      for (size_t i = 0; i < 6; i++)
         s[i] = i;
      for (size_t i = 0; i < 6; i++)
         cout << s[i] << "\n";
   }

   {
      print_line();
      cout << "\nTensor with static storage\n";
      stensor<float, 2, 3> a;
      using tensor_type = decltype(a);
      static_assert(std::is_same_v<tensor_type::storage_type,
                                   sdense_storage<float, 6>>);
      cout << "Rank = " << a.rank() << endl;
      cout << "Order = "
           << (a.storage_order() == storage_order::row_major ? "row_major"
                                                          : "col_major")
           << endl;
      cout << "is_vector = " << a.is_vector() << endl;
      cout << "is_matrix = " << a.is_matrix() << endl;
      cout << "is_transposed = " << a.is_transposed() << endl;
   }

   {
      print_line();
      using namespace ten::details;

      using A = make_dynamic_sequence<1>;
      static_assert(std::is_same_v<A::type::shape_type, shape<dynamic>>);

      using dA = dynamic_shape<1>;
      static_assert(std::is_same_v<dA, shape<dynamic>>);

      using B = make_dynamic_sequence<2>;
      static_assert(
          std::is_same_v<B::type::shape_type, shape<dynamic, dynamic>>);

      using dB = dynamic_shape<2>;
      static_assert(std::is_same_v<dB, shape<dynamic, dynamic>>);

      using C = make_dynamic_sequence<3>;
      static_assert(std::is_same_v<C::type::shape_type,
                                   shape<dynamic, dynamic, dynamic>>);

      using dC = dynamic_shape<3>;
      static_assert(std::is_same_v<dC, shape<dynamic, dynamic, dynamic>>);
   }

   {
      print_line();
      cout << "\nTensor shape and static strides\n";
      shape<2, 3, 4> s;
      using S = decltype(s);

      using stride_t = stride<S, storage_order::col_major>;
      cout << "Stride_0 = " << stride_t::template static_dim<0>() << endl;
      cout << "Stride_1 = " << stride_t::template static_dim<1>() << endl;
      cout << "Stride_2 = " << stride_t::template static_dim<2>() << endl;

      cout << S::static_dim<0>() << "x" << S::static_dim<1>() << "x"
           << S::static_dim<2>() << endl;
      cout << details::nth_static_stride<0, S>::value << endl;
      cout << details::nth_static_stride<1, S>::value << endl;
      cout << details::nth_static_stride<2, S>::value << endl;

      auto v = details::compute_static_strides<S, storage_order::col_major>();
      cout << "v = {";
      cout << v[0] << ", " << v[1] << ", " << v[2] << "}" << endl;
   }

   {
      print_line();
      cout << "\nDynamic strides\n";
      dynamic_shape<3> shape({2, 3, 4});
      stride<dynamic_shape<3>, storage_order::col_major> stride(shape);
      for (size_t i = 0; i < 3; i++)
         cout << stride.dim(i) << endl;
   }

   {
      print_line();
      cout << "\nTensor of dynamic shape\n";
      tensor<float, 2> A({3, 4});
      using Tensor_t = decltype(A);
      using Shape_t = Tensor_t::shape_type;
      const Tensor_t::shape_type &shape = A.shape();

      cout << "Rank = " << Shape_t::rank() << endl;
      cout << "Dynamic shape = [";
      cout << shape.dim(0) << ", ";
      cout << shape.dim(1) << "]\n";
      cout << "Is dynamic dims = \n";
      cout << Shape_t::is_dynamic_dim<0>() << endl;
      cout << Shape_t::is_dynamic_dim<1>() << endl;
      cout << "Dynamic size = " << shape.size() << endl;
      cout << "Must be uninit\n";
      using Node_t = Tensor_t::node_type;
      std::weak_ptr<Node_t> s = A.node();
      cout << A.node().get() << endl;
      auto p = s.lock();
      cout << p.get() << endl;
   }

   {
      print_line();
      cout << "Acess elements" << endl;
      tensor<float, 1> a({4});
      auto n = a.node();
      for (size_t i = 0; i < 4; i++)
         a[i] = i;
      for (size_t i = 0; i < 4; i++)
         cout << a[i] << endl;
   }

   {
      print_line();
      cout << "\nStatic Tensor - uninitialized memory\n";
      svector<float, 10> x;
      float *data = x.data();
      for (size_t i = 0; i < 10; i++)
         data[i] = float(i);
      for (size_t i = 0; i < 10; i++) {
         cout << "x[" << i << "] = " << data[i] << endl;
      }
   }

   {
      print_line();
      cout << "\nStatic Matrix\n";
      smatrix<float, 2, 3> x;
      float *data = x.data();
      for (size_t i = 0; i < 6; i++)
         data[i] = float(i);
      for (size_t i = 0; i < 2; i++) {
         for (size_t j = 0; j < 3; j++) {
            cout << "x(" << i << "," << j << ") = " << x(i, j) << endl;
         }
      }
   }

   {
      print_line();
      cout << "fill static tensor" << endl;
      auto x = fill<svector<float, 10>>(3.);
      cout << "x = ";
      for (size_t i = 0; i < 10; i++) {
         cout << x[i] << " ";
      }
      cout << endl;

      auto y = fill<float, shape<10>>(3.);
      cout << "y = ";
      for (size_t i = 0; i < 10; i++) {
         cout << y[i] << " ";
      }
      cout << endl;
   }

   {
      print_line();
      cout << "fill dynamic tensor" << endl;
      auto x = fill<ranked_tensor<float, shape<2, dynamic>>>(
          shape<2, dynamic>({2, 3}), 3.);
      cout << "x = ";
      for (size_t i = 0; i < 6; i++) {
         cout << x[i] << " ";
      }
      cout << endl;

      auto y = fill<ranked_tensor<float, shape<2, dynamic>>>({2, 3}, 3.);
      cout << "y = ";
      for (size_t i = 0; i < 6; i++) {
         cout << y[i] << " ";
      }
      cout << endl;

      auto z = fill<float, shape<2, dynamic>>({2, 3}, 3.);
      cout << "z = ";
      for (size_t i = 0; i < 6; i++) {
         cout << z[i] << " ";
      }
      cout << endl;
   }

   {
      print_line();
      cout << "zeros static tensor" << endl;
      auto x = zeros<ranked_tensor<float, shape<10>>>();
      cout << "x = ";
      for (size_t i = 0; i < 10; i++) {
         cout << x[i] << " ";
      }
      cout << endl;

      auto y = zeros<float, shape<10>>();
      cout << "y = ";
      for (size_t i = 0; i < 10; i++) {
         cout << y[i] << " ";
      }
      cout << endl;
   }

   {
      print_line();
      cout << "zeros dynamic tensor" << endl;
      auto x = zeros<ranked_tensor<float, shape<2, dynamic>>>(
          shape<2, dynamic>({2, 3}));
      cout << "x = ";
      for (size_t i = 0; i < 6; i++) {
         cout << x[i] << " ";
      }
      cout << endl;

      auto y = zeros<ranked_tensor<float, shape<2, dynamic>>>({2, 3});
      cout << "y = ";
      for (size_t i = 0; i < 6; i++) {
         cout << y[i] << " ";
      }
      cout << endl;

      auto z = zeros<float, shape<2, dynamic>>({2, 3});
      cout << "z = ";
      for (size_t i = 0; i < 6; i++) {
         cout << z[i] << " ";
      }
      cout << endl;
   }

   {
      print_line();
      cout << "ones static tensor" << endl;
      auto x = ones<svector<float, 10>>();
      cout << "x = ";
      for (size_t i = 0; i < 10; i++) {
         cout << x[i] << " ";
      }
      cout << endl;

      auto y = ones<float, shape<10>>();
      cout << "y = ";
      for (size_t i = 0; i < 10; i++) {
         cout << y[i] << " ";
      }
      cout << endl;
   }

   {
      print_line();
      cout << "ones dynamic tensor" << endl;
      auto x = ones<ranked_tensor<float, shape<2, dynamic>>>(
          shape<2, dynamic>({2, 3}));
      cout << "x = ";
      for (size_t i = 0; i < 6; i++) {
         cout << x[i] << " ";
      }
      cout << endl;

      auto y = ones<ranked_tensor<float, shape<2, dynamic>>>({2, 3});
      cout << "y = ";
      for (size_t i = 0; i < 6; i++) {
         cout << y[i] << " ";
      }
      cout << endl;

      auto z = ones<float, shape<2, dynamic>>({2, 3});
      cout << "z = ";
      for (size_t i = 0; i < 6; i++) {
         cout << z[i] << " ";
      }
      cout << endl;
   }

   {
      print_line();
      cout << "range static tensor" << endl;
      auto x = range<svector<float, 10>>();
      cout << "x = ";
      for (size_t i = 0; i < 10; i++) {
         cout << x[i] << " ";
      }
      cout << endl;

      auto y = range<float, shape<10>>();
      cout << "y = ";
      for (size_t i = 0; i < 10; i++) {
         cout << y[i] << " ";
      }
      cout << endl;
   }

   {
      print_line();
      cout << "range dynamic tensor" << endl;
      auto x = range<ranked_tensor<float, shape<2, dynamic>>>(
          shape<2, dynamic>({2, 3}));
      cout << "x = ";
      for (size_t i = 0; i < 6; i++) {
         cout << x[i] << " ";
      }
      cout << endl;

      auto y = range<ranked_tensor<float, shape<2, dynamic>>>({2, 3});
      cout << "y = ";
      for (size_t i = 0; i < 6; i++) {
         cout << y[i] << " ";
      }
      cout << endl;

      auto z = range<float, shape<2, dynamic>>({2, 3});
      cout << "z = ";
      for (size_t i = 0; i < 6; i++) {
         cout << z[i] << " ";
      }
      cout << endl;
   }

   {
      print_line();
      cout << "fill, zeros, ones and range for default float tensors" << endl;
      auto x = fill<matrix<float>>({2, 5}, 3.);
      cout << "x = ";
      for (size_t i = 0; i < 10; i++) {
         cout << x[i] << " ";
      }
      cout << endl;

      auto y = zeros<matrix<float>>({2, 5});
      cout << "y = ";
      for (size_t i = 0; i < 10; i++) {
         cout << y[i] << " ";
      }
      cout << endl;

      auto z = ones<matrix<float>>({2, 5});
      cout << "z = ";
      for (size_t i = 0; i < 10; i++) {
         cout << z[i] << " ";
      }
      cout << endl;

      auto t = range<matrix<float>>({2, 5});
      cout << "t = ";
      for (size_t i = 0; i < 10; i++) {
         cout << t[i] << " ";
      }
      cout << endl;
   }

   {
      print_line();
      cout << "Reshape static tensor" << endl;
      auto x = range<smatrix<float, 2, 3>>();
      auto y = reshape<shape<3, 2>>(x).eval();
      print_tensor(x);
      print_tensor(y);
      y[1] = 99.;
      print_tensor(x);
      print_tensor(y);
      auto z = reshape<3, 2>(x).eval();
      print_tensor(z);
   }

   {
      print_line();
      cout << "Reshape dynamic tensor" << endl;
      auto x = range<matrix<float>>({2, 3});
      auto y = reshape(x, shape<dynamic, dynamic>({3, 2})).eval();
      print_tensor(x);
      print_tensor(y);
      y[1] = 99.;
      print_tensor(x);
      print_tensor(y);
      auto z = reshape<2>(x, {3, 2}).eval();
      print_tensor(z);
   }

   {
      print_line();
      cout << "Uninitialized dynamic vector" << endl;
      ten::vector<float> x(5);
      print_tensor(x);
   }

   {
      print_line();
      cout << "Uninitialized dynamic matrix" << endl;
      matrix<float> x(2, 3);
      print_tensor(x);
   }

   {
      print_line();
      cout << "col_major tensor" << endl;
      tensor<float, 3, storage_order::col_major> x({2, 3, 4});
      cout << "shape = " << x.shape() << endl;
      cout << "strides = " << x.strides() << endl;
   }

   {
      print_line();
      cout << "row_major tensor" << endl;
      tensor<float, 3, storage_order::row_major> x({2, 3, 4});
      cout << "shape = " << x.shape() << endl;
      cout << "strides = " << x.strides() << endl;
   }

   return 0;
}
