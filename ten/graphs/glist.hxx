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

/// Graph as an unweighted adjacency list
template<class T>
class glist {
 private:
   ten::graph::graph_type _type;
   std::map<T, std::vector<T>> _graph;

 public:
   glist(ten::graph::graph_type graph_type = ten::graph::graph_type::undirected)
       : _type(graph_type) {}

   bool empty() { return _graph.empty(); }

   void add_vertex(const T& vertex) { _graph[vertex] = std::vector<T>(); }

   void add_edge(const T& src, const T& dest) {
      _graph[src].push_back(dest);
      if (_type == graph_type::undirected) {
         _graph[dest].push_back(src);
      }
   }

   bool has_edge(const T& src, const T& dest) {
      return std::find(_graph[src].begin(), _graph[src].end(), dest) !=
             _graph[src].end();
   }

   // Get the adjacency matrix
   auto matrix() -> ten::matrix<float> {
      size_t n = _graph.size();
      ten::matrix<float> m = ten::zeros<ten::matrix<float>>({n, n});
      std::map<T, size_t> map;
      size_t i = 0;
      for (auto it = _graph.begin(); it != _graph.end(); it++) {
         map[it->first] = i;
         i++;
      }
      for (auto const &[src, value] : _graph) {
         for (auto dest : value) {
            m(map[src], map[dest]) = 1.;
         }
      }
      return m;
   }

   void dfs(const T& u, std::function<void(const T&)> f) {
      std::stack<T> stack;
      stack.push(u);
      std::unordered_set<T> visited;
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

   void bfs(const T& u, std::function<void(const T&)> f) {
      std::queue<T> queue;
      queue.push(u);
      std::unordered_set<T> visited;
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
