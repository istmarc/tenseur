#include "ten/graph_theory/gmatrix.hxx"
#include <ten/tensor>
#include <ten/graph>

int main() {
   {
      ten::graph::gmatrix<> g(5);
      g.add_edge(0, 1);
      g.add_edge(1, 0);
      g.add_edge(0, 4);
      g.add_edge(4, 0);
      g.add_edge(1, 4);
      g.add_edge(4, 1);
      g.add_edge(1, 2);
      g.add_edge(2, 1);
      g.add_edge(2, 3);
      g.add_edge(3, 2);
      g.add_edge(4, 3);
      g.add_edge(3, 4);
      std::cout << g << std::endl;
   }

   {
      ten::graph::gmatrix<float> g(5);
      g.add_edge(0, 1);
      g.add_edge(1, 0);
      g.add_edge(0, 4);
      g.add_edge(4, 0);
      g.add_edge(1, 4);
      g.add_edge(4, 1);
      g.add_edge(1, 2);
      g.add_edge(2, 1);
      g.add_edge(2, 3);
      g.add_edge(3, 2);
      g.add_edge(4, 3);
      g.add_edge(3, 4);
      std::cout << g << std::endl;
   }

}
