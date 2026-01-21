#include <ten/tensor>

#include <memory>

namespace ten::nn {

/// Dense/Linear layer
/// y = w * x + b
/// where :
///   - w of shape {in_features x out_features}
///   - b of shape {out_features} and
///   - x of shape {out_features}
template <class T = float> struct dense {
   using MatrixType = ten::matrix<T>;
   using VectorType = ten::vector<T>;
   std::shared_ptr<MatrixType> _w = nullptr;
   std::shared_ptr<VectorType> _b = nullptr;

   auto weights() const { return _w; }

   auto bias() const { return _b; }

   dense(size_t in_features, size_t out_features) {
      auto wshape = ten::shape<0, 0>({in_features, out_features});
      _w = std::make_shared<MatrixType>(wshape, true);
      // Initialize with Uniform[-sqrt(k), sqrt(k)] where k = 1 / in_features
      T k = std::sqrt(1 / T(in_features));
      ten::uniform<T> unif(-k, k);
      for (size_t i = 0; i < _w->size(); i++) {
         (*_w.get())[i] = unif.sample();
      }
      auto bshape = ten::shape<0>({out_features});
      _b = std::make_shared<VectorType>(bshape, true);
      // Initialize with Uniform[-sqrt(k), sqrt(k)] where k = 1 / in_features
      for (size_t i = 0; i < _b->size(); i++) {
         (*_b.get())[i] = unif.sample();
      }
   }

   template <Expr ExprType> auto operator()(ExprType &x) {
      return (*_w.get()) * x + (*_b.get());
   }

   template <Expr ExprType> auto forward(ExprType &x) { return (*this)(x); }
};

/// Dense/Linear layer
/// y = X * W + B
/// where :
/// - X of shape {batch x in_features}
/// - W of shape {in_features x out_features}
/// - B of shape {batch x out_features}
/// - Y of shape {batch x out_features}
template <class T = float> struct batched_dense {
   using MatrixType = ten::matrix<T>;
   std::shared_ptr<MatrixType> _w = nullptr;
   std::shared_ptr<MatrixType> _b = nullptr;

   auto weights() const { return _w; }

   auto bias() const { return _b; }

   batched_dense(size_t batch, size_t in_features, size_t out_features) {
      auto wshape = ten::shape<0, 0>({in_features, out_features});
      _w = std::make_shared<MatrixType>(wshape, true);
      // Initialize with Uniform[-sqrt(k), sqrt(k)] where k = 1 / in_features
      T k = std::sqrt(1 / T(in_features));
      ten::uniform<T> unif(-k, k);
      for (size_t i = 0; i < _w->size(); i++) {
         (*_w.get())[i] = unif.sample();
      }
      auto bshape = ten::shape<0, 0>({batch, out_features});
      _b = std::make_shared<MatrixType>(bshape, true);
      // Initialize with Uniform[-sqrt(k), sqrt(k)] where k = 1 / in_features
      for (size_t i = 0; i < _b->size(); i++) {
         (*_b.get())[i] = unif.sample();
      }
   }

   template <Expr ExprType> auto operator()(ExprType &x) {
      return x * (*_w.get()) + (*_b.get());
   }

   template <Expr ExprType> auto forward(ExprType &x) { return (*this)(x); }
};

} // namespace ten::nn
