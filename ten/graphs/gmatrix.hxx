#ifndef TENSEUR_GRAPHS_GMATRIX
#define TENSEUR_GRAPHS_GMATRIX

#include <ten/tensor>
#include <ten/io>
#include <ten/graphs/types.hxx>

namespace ten{
namespace graph{

template<class T = float>
class gmatrix{
   private:
      size_t _vertices;
      ten::matrix<T> _m;
      graph_type _gtype;

   public:
      gmatrix(const size_t vertices, const graph_type gtype = graph_type::undirected):
         _vertices(vertices), _gtype(gtype) {
         _m = ten::zeros<ten::matrix<T>>({vertices, vertices});
      }

   void add_edge(size_t src, size_t dest, const T value = 1.0) {
      _m(src, dest) = value;
      if (_gtype == graph_type::undirected) {
         _m(dest, src) = value;
      }
   }

   bool has_edge(size_t src, size_t dest) const {
      return _m(src, dest) != 0.;
   }

   template<class _T>
   friend std::ostream &operator<<(std::ostream&, const gmatrix<_T>& );
};

template<class T>
std::ostream &operator<<(std::ostream& os, const gmatrix<T>& g) {
   os << g._m;
   return os;
}

}
}


#endif
