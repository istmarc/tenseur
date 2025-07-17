#ifndef TENSEUR_GRAPH_THEORY_GLIST
#define TENSEUR_GRAPH_THEORY_GLIST

#include <ten/tensor>

#include <ten/graphs/types.hxx>

#include <map>
#include <queue>
#include <stack>
#include <unordered_set>

namespace ten {
namespace graph {

/// Grpah as an unweighted adjacency list
/// Support vertices from 0...n
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
      ten::matrix<float> m = ten::zeros<ten::matrix<float>>({n, n});
      for (auto const &[src, value] : _graph) {
         for (auto dest : value) {
            m(src, dest) = 1.;
         }
      }
      return m;
   }

   void dfs(const size_t u, std::function<void(const size_t)> f) {
      std::stack<size_t> stack;
      stack.push(u);
      std::unordered_set<size_t> visited;
      visited.insert(u);

      while (!stack.empty()) {
         auto v = stack.top();
         stack.pop();
         f(v);
         for (auto n : _graph[v]) {
            if (visited.find(n) == visited.end()) {
               visited.insert(n);
               stack.push(n);
            }
         }
      }
   }

   void bfs(size_t u, std::function<void(const size_t)> f) {
      std::queue<size_t> queue;
      queue.push(u);
      std::unordered_set<size_t> visited;
      visited.insert(u);

      while (!queue.empty()) {
         auto v = queue.front();
         queue.pop();
         f(v);
         for (auto n : _graph[v]) {
            if (visited.find(n) == visited.end()) {
               visited.insert(n);
               queue.push(n);
            }
         }
      }
   }
};

} // namespace graph
} // namespace ten

#endif
