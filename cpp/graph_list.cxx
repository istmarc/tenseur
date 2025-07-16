#include <ten/graph>
#include <ten/tensor>

int main() {

   {
      auto g = ten::graph::glist();
      size_t n = 5;
      for (size_t i = 0; i < n; i++) {
         g.add_vertex(i);
      }
      for (size_t i = 0; i < n; i++) {
         for (size_t j = i + 1; j < n; j++) {
            g.add_edge(i, j);
         }
      }
      std::cout << "Adjacency matrix" << std::endl;
      std::cout << g.matrix() << std::endl;
   }

   {
      auto g = ten::graph::gweighted();
      size_t n = 5;
      for (size_t i = 0; i < n; i++) {
         g.add_vertex(i);
      }
      for (size_t i = 0; i < n; i++) {
         for (size_t j = i + 1; j < n; j++) {
            g.add_edge(i, j, i * j);
         }
      }
      std::cout << "Adjacency matrix" << std::endl;
      std::cout << g.matrix() << std::endl;
   }
}
