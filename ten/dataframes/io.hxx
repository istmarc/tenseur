#ifndef TENSEUR_DATAFRAMES_IO
#define TENSEUR_DATAFRAMES_IO

#include <fstream>
#include <iostream>
#include <string>
#include <cctype>

#include <ten/dataframes/dataframe.hxx>

namespace ten{

bool isdigits(const std::string& s) {
   bool res = true;
   for (size_t i = 0;i < s.size(); i++) {
      if (!std::isdigit(s[i])) {
         res = false;
         break;
      }
   }
   return res;
}

bool isreal(const std::string& s) {
   bool res = true;
   for (size_t i = 0;i < s.size(); i++) {
      if ((!std::isdigit(s[i])) && (s[i] != '.')) {
         res = false;
         break;
      }
   }
   auto it = std::find(s.begin(), s.end(), '.');
   return res && (it != s.end());
}

struct csv_options{
   char sep = ',';
   bool index = true;
   bool header = true;
};


// Read from csv to dataframe
dataframe read_csv(const std::string& filename, const csv_options& options = csv_options()) {
   dataframe df;
   std::ifstream fs(filename);
   if (!fs.is_open()) {
      std::cout << "Failed to open file " << filename << '\n';
   } else {
      // Read the number of rows
      std::string line;
      size_t rows = 0;
      size_t cols = 1;
      // Read first line
      std::getline(fs, line);
      for (auto it = line.begin(); it != line.end(); it++) {
         if (*it == options.sep) {
            cols++;
         }
      }
      if (options.header) {
         std::getline(fs, line);
      }
      // Infer types from the first line
      std::vector<data_type> types({cols});
      size_t col_pos = 0;
      for (size_t j = 0; j < cols; j++) {
         long sep_pos = line.find(options.sep, col_pos+1);
         if (sep_pos == -1) {
            sep_pos = line.size();
         }
         std::string s = line.substr(col_pos, sep_pos - col_pos);
         if (isdigits(s)) {
            types[j] = data_type::int64;
         } else if (isreal(s)) {
            types[j] = data_type::float32;
         } else {
            types[j] = data_type::string;
         }
         col_pos = sep_pos + 1;
      }
      while (std::getline(fs, line)) {
         rows++;
      }
      fs.close();
      // Dataset
      std::vector<std::vector<cell_type>> data({cols});
      std::vector<std::string> names(cols);
      fs.open(filename);
      if (options.header) {
         std::getline(fs, line);
         size_t pos = 0;
         for (size_t i = 0; i < cols; i++) {
            long next_pos = line.find(options.sep, pos+1);
            if (next_pos == -1) {
               names[i] = std::string(line.substr(pos, line.size()));
            } else {
               names[i] = std::string(line.substr(pos, next_pos - pos + 1));
            }
         }
      } else {
         for (size_t i = 0; i < cols; i++) {
            names[i] = std::to_string(i);
         }
      }

      for (size_t i = 0; i < rows; i++) {
         std::getline(fs, line);
         size_t pos = 0;
         for (size_t j = 0; j < cols; j++) {
            long sep_pos = line.find(options.sep, pos+1);
            if (sep_pos == -1) {
               sep_pos = line.size();
            }
            std::string s = line.substr(pos, sep_pos - pos);
            if (types[j] == data_type::boolean) {
               bool x = (std::stoi(s) == 1);
               data[j].push_back(x);
            } else if (types[j] == data_type::int32) {
               int32_t x = std::stoi(s);
               data[j].push_back(x);
            } else if (types[j] == data_type::uint32) {
               uint32_t x = std::stoi(s);
               data[j].push_back(x);
            } else if (types[j] == data_type::int64) {
               int64_t x = std::stol(s);
               data[j].push_back(x);
            } else if (types[j] == data_type::uint64) {
               uint64_t x = std::stoul(s);
               data[j].push_back(x);
            } else if (types[j] == data_type::float32) {
               float x = std::stof(s);
               data[j].push_back(x);
            } else if (types[j] == data_type::float64) {
               double x = std::stod(s);
               data[j].push_back(x);
            } else if (types[j] == data_type::string) {
            data[j].push_back(std::string(s));
            }
         pos = sep_pos + 1;
         }
      }

      // Add data to the data frame
      for (size_t i = 0; i < cols; i++) {
         df.add_col(names[i], types[i], data[i]);
      }

      // Close the file
      fs.close();
   }

   return df;
}

void write_csv(const dataframe& df, const std::string& filename, const csv_options& options = csv_options()) {
   std::ofstream fs(filename);
   if (!fs.is_open()) {
      std::cout << "Failed to open file " << filename << "\n";
   } else {
      size_t rows = df.rows();
      size_t cols = df.cols();
      // Write the names
      auto names = df.names();
      auto indices = df.indices();
      if (options.index) {
         fs << ',';
      }
      for (size_t i = 0; i < cols; i++) {
         fs << names[i];
         if (i+1 < cols) {
            fs << options.sep;
         }
      }
      fs << '\n';
      // Write the data
      for (size_t i = 0; i < rows; i++) {
         if (options.index) {
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
               fs << options.sep;
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
   if (index == -1) {
      filename.append(".ten");
   }
   auto ext = filename.substr(index + 1, filename.size());
   if (ext == "ten") {
      std::cout << "Saving to binary file not implemented" << std::endl;
      return;
   } else if (ext == "csv") {
      write_csv(df, filename);
   }
}

}

#endif
