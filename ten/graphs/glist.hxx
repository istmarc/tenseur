#ifndef TENSEUR_GRAPH_THEORY_GLIST
#define TENSEUR_GRAPH_THEORY_GLIST

#include <ten/tensor>

#include <ten/graphs/types.hxx>

#include <map>

namespace ten {
namespace graph {

/// Grpah as an unweighted adjacency list
class glist {
 private:
   ten::graph::graph_type _type;
   std::map<size_t, std::vector<size_t>> _graph;

 public:
   glist(ten::graph::graph_type graph_type = ten::graph::graph_type::undirected)
       : _type(graph_type) {}

   bool empty() { return _graph.empty(); }

   void add_vertex(size_t vertex) { _graph[vertex] = std::vector<size_t>(); }

   void add_edge(size_t src, size_t dest) {
      _graph[src].push_back(dest);
      if (_type == graph_type::undirected) {
         _graph[dest].push_back(src);
      }
   }

   bool has_edge(size_t src, size_t dest) {
      return std::find(_graph[src].begin(), _graph[src].end(), dest) !=
             _graph[src].end();
   }

   // Get the adjacency matri
   auto matrix() -> ten::matrix<float> {
      size_t n = _graph.size();
      ten::matrix<float> m({n, n});
      for (size_t i = 0; i < n; i++) {
         for (size_t j = 0; j < n; j++) {
            m(i, j) = has_edge(i, j);
         }
      }
      return m;
   }
};

} // namespace graph
} // namespace ten

#endif
