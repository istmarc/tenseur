#ifndef TENSEUR_DATAFRAMES_IO
#define TENSEUR_DATAFRAMES_IO

#include <fstream>
#include <iostream>
#include <string>

#include <ten/dataframes/dataframe.hxx>

namespace ten{

dataframe read_csv(const std::string& filename, char sep = ',', bool index = true) {
   dataframe df;
   std::fstream fs(filename);
   if (!fs.is_open()) {
      std::cout << "Failed to open file " << filename << "\n";
   } else {
      // Read the number of rows
      std::string line;
      size_t rows = 0;
      while (std::getline(fs, line)) {
         rows++;
      }
      std::cout << "ROws = " << rows << std::endl;
   }

   return df;
}

void write_csv(const dataframe& df, const std::string& filename, char sep = ',',bool index = true) {
   std::ofstream fs(filename);
   if (!fs.is_open()) {
      std::cout << "Failed to open file " << filename << "\n";
   } else {
      std::cout << "Writing csv file" << std::endl;
      size_t rows = df.rows();
      size_t cols = df.cols();
      // Write the names
      auto names = df.names();
      auto indices = df.indices();
      if (index) {
         fs << ',';
      }
      for (size_t i = 0; i < cols; i++) {
         fs << names[i];
         if (i+1 < cols) {
            fs << sep;
         }
      }
      fs << '\n';
      // Write the data
      for (size_t i = 0; i < rows; i++) {
         if (index) {
            data_type type = df.indices_type();
            cell_type idx = df.index(i);
            if (type == data_type::boolean) {
               bool x = std::get<bool>(idx);
               fs << std::to_string(x);
            }
            if (type == data_type::int32) {
               int32_t x = std::get<int32_t>(idx);
               fs << std::to_string(x);
            }
            if (type == data_type::int64) {
               int64_t x = std::get<int64_t>(idx);
               fs << std::to_string(x);
            }
            if (type == data_type::uint32) {
               uint32_t x = std::get<uint32_t>(idx);
               fs << std::to_string(x);
            }
            if (type == data_type::uint64) {
               uint64_t x = std::get<uint64_t>(idx);
               fs << std::to_string(x);
            }
            if (type == data_type::float32) {
               float x = std::get<float>(idx);
               fs << std::to_string(x);
            }
            if (type == data_type::float64) {
               double x = std::get<double>(idx);
               fs << std::to_string(x);
            }
            if (type == data_type::string) {
               std::string x = std::get<std::string>(idx);
               fs << x;
            }
            fs << ',';
         }
         for (size_t j = 0; j < cols; j++) {
            cell_type value = df(i, j);
            data_type type = df.type(j);
            if (type == data_type::boolean) {
               bool x = std::get<bool>(value);
               fs << std::to_string(x);
            }
            if (type == data_type::int32) {
               int32_t x = std::get<int32_t>(value);
               fs << std::to_string(x);
            }
            if (type == data_type::int64) {
               int64_t x = std::get<int64_t>(value);
               fs << std::to_string(x);
            }
            if (type == data_type::uint32) {
               uint32_t x = std::get<uint32_t>(value);
               fs << std::to_string(x);
            }
            if (type == data_type::uint64) {
               uint64_t x = std::get<uint64_t>(value);
               fs << std::to_string(x);
            }
            if (type == data_type::float32) {
               float x = std::get<float>(value);
               fs << std::to_string(x);
            }
            if (type == data_type::float64) {
               double x = std::get<double>(value);
               fs << std::to_string(x);
            }
            if (type == data_type::string) {
               std::string x = std::get<std::string>(value);
               fs << x;
            }
            if (j + 1 < cols) {
               fs << sep;
            } else {
               fs << '\n';
            }
         }
      }
   }
}

// Save data frame to file
void save(const dataframe& df, std::string filename) {
   long index = filename.rfind('.');
   std::cout << "pos = " << index << std::endl;
   if (index == -1) {
      filename.append(".ten");
   }
   auto ext = filename.substr(index + 1, filename.size());
   std::cout << "ext = " << ext << std::endl;
   if (ext == "ten") {
      std::cout << "Saving to binary file not implemented" << std::endl;
      return;
   } else if (ext == "csv") {
      write_csv(df, filename);
   }
}

}

#endif
