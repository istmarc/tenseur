#ifndef TENSEUR_GRAPH_THEORY_GWEIGHTED
#define TENSEUR_GRAPH_THEORY_GWEIGHTED

#include <ten/tensor>

#include <ten/graphs/types.hxx>
#include <map>

namespace ten {
namespace graph {

/// Graph as a weighted ajanceny list
template<class T = float>
class gweighted {
 private:
   ten::graph::graph_type _type;
   std::map<size_t, std::vector<std::pair<size_t, T>>> _graph;

 public:
   gweighted(ten::graph::graph_type graph_type = ten::graph::graph_type::undirected)
       : _type(graph_type) {}

   bool empty() { return _graph.empty(); }

   void add_vertex(size_t vertex) { _graph[vertex] = std::vector<std::pair<size_t, T>>(); }

   void add_edge(size_t src, size_t dest, T weight) {
      _graph[src].push_back(std::make_pair(dest, weight));
      if (_type == graph_type::undirected) {
         _graph[dest].push_back(std::make_pair(src, weight));
      }
   }

   bool has_edge(size_t src, size_t dest) {
      auto it = _graph[src].begin();
      while (it < _graph[src].end()) {
         auto [d, w] = *it;
         if (d == dest) return true;
         it++;
      }
      return false;
   }

   // Get the weight of the edge src->dest
   T weight(size_t src, size_t dest) {
      auto it = _graph[src].begin();
      while (it < _graph[src].end()) {
         auto [d, w] = *it;
         if (d == dest) return w;
         it++;
      }
      return T(0);
   }

   // Get the adjacency matrix
   auto matrix() -> ten::matrix<T> {
      size_t n = _graph.size();
      ten::matrix<float> m({n, n});
      for (size_t i = 0; i < n; i++) {
         for (size_t j = 0; j < n; j++) {
            m(i, j) = weight(i, j);
         }
      }
      return m;
   }
};

} // namespace graph
} // namespace ten

#endif
