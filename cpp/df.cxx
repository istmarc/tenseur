#include <ten/tensor>
#include <ten/dataframe>

int main() {
   ten::dataframe df;

   ten::vector<float> x({3}, {1.0f, 2.0f, 3.0f});
   std::vector<std::string> y({"A", "B", "C"});
   ten::vector<double> z({3}, {-1.0, -2.0, -3.0});

   df.add_col("x", x);
   df.add_col("y", y);
   df.add_col("z", z);

   std::cout << df << std::endl;

   {
      auto s = df[{"x"}];
   }

}

