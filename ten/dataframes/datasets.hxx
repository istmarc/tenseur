#include <iostream>
#include <filesystem>

#include <ten/dataframes/dataframe.hxx>
#include <ten/dataframes/io.hxx>

namespace ten::datasets {

// Load iris dataset
dataframe load_iris(const std::string& path) {
   return read_csv(path + "/ten/datasets/iris.csv", csv_options{.index = false, .header = false});
}

}
