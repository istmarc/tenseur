#ifndef TENSEUR_GRAPH_THEORY_GMATRIX
#define TENSEUR_GRAPH_THEORY_GMATRIX

#include <ten/tensor>
#include <ten/graph_theory/types.hxx>

namespace ten{
namespace graph{

template<class T = bool>
class gmatrix{
   private:
      size_t _vertices;
      ten::matrix<T> _m;
      ten::graph_type _gtype;

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

template<>
class gmatrix<bool> {
   private:
      size_t _vertices;
      std::vector<bool> _m;
      graph_type _gtype;

   size_t get_index(size_t i, size_t j) const {
      return i * _vertices + j;
   }

   public:
      gmatrix(const size_t vertices,const graph_type gtype = graph_type::undirected): 
         _vertices(vertices), _gtype(gtype) {
         _m = std::vector<bool>(vertices * vertices * vertices, false);
      };

   void add_edge(size_t src, size_t dest) {
      size_t idx = get_index(src, dest);
      _m[idx] = true;
      if (_gtype == graph_type::undirected) {
         size_t idx = get_index(dest, src);
         _m[idx] = true;
      }
   }

   bool has_edge(size_t src, size_t dest) const {
      size_t idx = get_index(src, dest);
      return _m[idx];
   }

   template<class _T>
   friend std::ostream &operator<<(std::ostream&, const gmatrix<_T>& );
};

template<>
std::ostream &operator<<(std::ostream& os, const gmatrix<bool>& g) {
   os << "matrix<bool," << g._vertices << "x" << g._vertices << ">\n";
   for (size_t i = 0; i < g._vertices; i++) {
      os << g.has_edge(i, 0);
      for (size_t j = 1; j < g._vertices; j++) {
         os << "   " << g.has_edge(i, j);
      }
      os << "\n";
   }
   return os;
}

}
}


#endif
