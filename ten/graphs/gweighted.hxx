#ifndef TENSEUR_GRAPH_THEORY_GWEIGHTED
#define TENSEUR_GRAPH_THEORY_GWEIGHTED

#include <ten/tensor>

#include <map>
#include <ten/graphs/types.hxx>

namespace ten {
namespace graph {

/// Graph as a weighted ajanceny list
template <class T, class W = float> class gweighted {
 private:
   ten::graph::graph_type _type;
   std::map<T, std::vector<std::pair<T, W>>> _graph;

 public:
   gweighted(
       ten::graph::graph_type graph_type = ten::graph::graph_type::undirected)
       : _type(graph_type) {}

   bool empty() { return _graph.empty(); }

   void add_vertex(const T &vertex) {
      _graph[vertex] = std::vector<std::pair<T, W>>();
   }

   void add_edge(const T &src, const T &dest, W weight) {
      _graph[src].push_back(std::make_pair(dest, weight));
      if (_type == graph_type::undirected) {
         _graph[dest].push_back(std::make_pair(src, weight));
      }
   }

   bool has_edge(const T &src, const T &dest) {
      auto it = _graph[src].begin();
      while (it < _graph[src].end()) {
         auto [d, w] = *it;
         if (d == dest)
            return true;
         it++;
      }
      return false;
   }

   // Get the weight of the edge src->dest
   T weight(const T &src, const T &dest) {
      auto it = _graph[src].begin();
      while (it < _graph[src].end()) {
         auto [d, w] = *it;
         if (d == dest)
            return w;
         it++;
      }
      return T(0);
   }

   // Get the adjacency matrix
   auto matrix() -> ::ten::matrix<W> {
      size_t n = _graph.size();
      ::ten::matrix<W> m = ::ten::zeros<::ten::matrix<W>>({n, n});
      std::map<T, size_t> map;
      size_t i = 0;
      for (auto it = _graph.begin(); it != _graph.end(); it++) {
         map[it->first] = i;
         i++;
      }

      for (auto const &[src, value] : _graph) {
         for (auto const &[dest, weight] : value) {
            m(map[src], map[dest]) = weight;
         }
      }
      return m;
   }
};

} // namespace graph
} // namespace ten

#endif
