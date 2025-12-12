#ifndef TENSEUR_DATAFRAME_HXX
#define TENSEUR_DATAFRAME_HXX

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <variant>

#include <ten/tensor>

namespace ten {

using cell_type = std::variant<bool, int32_t, uint32_t, int64_t, uint64_t,
                               float, double, std::string>;

// Data frame node
struct dataframe_node {
   using vector_type = std::vector<cell_type>;

   // Number of rows
   size_t _rows = 0;
   size_t _cols = 0;
   // Indices
   std::vector<cell_type> _indices;
   // Column names
   std::vector<std::string> _names;
   // Column types
   std::vector<data_type> _types;
   // Data column name -> column data
   std::map<std::string, std::shared_ptr<vector_type>> _data;

   cell_type &at(const std::string &name, size_t index) {
      auto ptr = _data[name].get();
      return (*ptr)[index];
   }

   // Return whether the column name exist or not
   bool has_column_name(const std::string &name) {
      for (size_t i = 0; i < _names.size(); i++) {
         if (_names[i] == name) {
            return true;
         }
      }
      return false;
   }

   // Return the index of the column name or -1
   long index_from_name(const std::string &name) {
      for (size_t i = 0; i < _names.size(); i++) {
         if (name == _names[i]) {
            return i;
         }
      }
      return -1;
   }

   // Return the column indices from names
   std::vector<size_t> column_indices(const std::vector<std::string> &names) {
      std::vector<size_t> indices;
      for (size_t i = 0; i < names.size(); i++) {
         long idx = index_from_name(names[i]);
         [[unlikely]] if (idx == -1) {
            continue;
         } else {
            indices.push_back(idx);
         }
      }
      return indices;
   }

   // Get the column names from indices
   std::vector<std::string> column_names(const std::vector<size_t> &indices) {
      std::vector<std::string> names;
      for (size_t i = 0; i < indices.size(); i++) {
         if (i < 0 || i >= _cols) {
            continue;
         } else {
            names.push_back(_names[i]);
         }
      }
      return names;
   }

   void remove_column(const size_t index) {
      // Remove from the column names
      size_t size = _cols;
      // Save the column names into names
      std::vector<std::string> names({size});
      for (size_t i = 0; i < size; i++) {
         names[i] = _names[i];
      }
      // Copy all the column names except name
      _names = std::vector<std::string>();
      for (size_t i = 0; i < size; i++) {
         if (i != index) {
            _names.push_back(names[i]);
         }
      }
   }

   void remove_types(const size_t index) {
      size_t size = _cols;
      // Save the types into types
      std::vector<data_type> types({size});
      for (size_t i = 0; i < size; i++) {
         types[i] = _types[i];
      }
      // Copy all the data types except the index'th data type
      _types = std::vector<data_type>();
      for (size_t i = 0; i < size; i++) {
         if (i != index) {
            _types.push_back(types[i]);
         }
      }
   }

   void remove_data(const std::string &name) { _data.erase(name); }
};

// Forward declaration of data frame
class dataframe;

// Data frame view
class dataframe_view {
 private:
   std::vector<size_t> _row_indices;
   std::vector<std::string> _col_names;
   std::vector<data_type> _types;
   std::shared_ptr<dataframe_node> _node;

 public:
   dataframe_view(std::shared_ptr<dataframe_node> node,
                  const std::vector<size_t> &row_indices,
                  const std::vector<std::string> &col_names)
       : _row_indices(row_indices), _col_names(col_names), _node(node) {
      // Get the column indices
      std::vector<size_t> indices = node->column_indices(_col_names);
      // Fill the types
      for (auto i : indices) {
         _types.push_back(node->_types[i]);
      }
   }

   // Assign a value
   template <typename T> dataframe_view &operator=(const T &value) {
      for (std::string name : _col_names) {
         for (size_t row : _row_indices) {
            _node->at(name, row) = value;
         }
      }
      return *this;
   }

   // Assign a ten::vector
   template <typename T>
   dataframe_view &operator=(const ten::vector<T> &values) {
      for (std::string name : _col_names) {
         size_t k = 0;
         for (size_t row : _row_indices) {
            _node->at(name, row) = values[k];
            k++;
         }
      }
      return *this;
   }

   // Assign a std::vector
   template <typename T>
   dataframe_view &operator=(const std::vector<T> &values) {
      for (std::string name : _col_names) {
         size_t k = 0;
         for (size_t row : _row_indices) {
            _node->at(name, row) = values[k];
            k++;
         }
      }
      return *this;
   }

   friend std::ostream &operator<<(std::ostream &os, const dataframe_view &df);
};

namespace {
std::string center_string(const std::string &s, size_t size) {
   if (size <= s.size()) {
      return s;
   }
   size_t n = s.size();
   size_t m = size - n;
   size_t k = m / 2;
   return std::string(k, ' ') + s + std::string(k + m % 2, ' ');
}
} // namespace

std::ostream &operator<<(std::ostream &os, const dataframe_view &df) {
   size_t cols = df._col_names.size();
   size_t rows = df._row_indices.size();
   os << "dataframe_view[" << rows << "x" << cols << "]\n";
   std::vector<size_t> max_len({cols});
   for (size_t i = 0; i < cols; i++) {
      std::string name = df._col_names[i];
      max_len[i] = name.size();
      data_type type = df._types[i];
      for (size_t j = 0; j < rows; j++) {
         if (type == data_type::boolean) {
            bool x = std::get<bool>(df._node->at(name, j));
            size_t len = std::to_string(x).size();
            max_len[i] = std::max(max_len[i], len);
         }
         if (type == data_type::int32) {
            int32_t x = std::get<int32_t>(df._node->at(name, j));
            size_t len = std::to_string(x).size();
            max_len[i] = std::max(max_len[i], len);
         }
         if (type == data_type::int64) {
            int64_t x = std::get<int64_t>(df._node->at(name, j));
            size_t len = std::to_string(x).size();
            max_len[i] = std::max(max_len[i], len);
         }
         if (type == data_type::uint32) {
            uint32_t x = std::get<uint32_t>(df._node->at(name, j));
            size_t len = std::to_string(x).size();
            max_len[i] = std::max(max_len[i], len);
         }
         if (type == data_type::uint64) {
            uint64_t x = std::get<uint64_t>(df._node->at(name, j));
            size_t len = std::to_string(x).size();
            max_len[i] = std::max(max_len[i], len);
         }
         if (type == data_type::float32) {
            float x = std::get<float>(df._node->at(name, j));
            size_t len = std::to_string(x).size();
            max_len[i] = std::max(max_len[i], len);
         }
         if (type == data_type::float64) {
            double x = std::get<double>(df._node->at(name, j));
            size_t len = std::to_string(x).size();
            max_len[i] = std::max(max_len[i], len);
         }
         if (type == data_type::string) {
            std::string x = std::get<std::string>(df._node->at(name, j));
            size_t len = x.size();
            max_len[i] = std::max(max_len[i], len);
         }
      }
   }
   // Print +----+
   os << "+";
   for (size_t i = 0; i < cols; i++) {
      for (size_t j = 0; j < max_len[i]; j++) {
         os << "-";
      }
      if (i + 1 < cols) {
         os << "+";
      }
   }
   os << "+\n";
   os << "|";
   for (size_t i = 0; i < cols; i++) {
      std::string name = df._col_names[i];
      os << center_string(name, max_len[i]);
      os << "|";
   }
   os << "\n";
   // Print +----+
   os << "+";
   for (size_t i = 0; i < cols; i++) {
      for (size_t j = 0; j < max_len[i]; j++) {
         os << "-";
      }
      if (i + 1 < cols) {
         os << "+";
      }
   }
   os << "+\n";
   // Print the values
   for (size_t j = 0; j < rows; j++) {
      os << "|";
      for (size_t i = 0; i < cols; i++) {
         std::string name = df._col_names[i];
         data_type type = df._types[i];
         if (type == data_type::boolean) {
            bool x = std::get<bool>(df._node->at(name, j));
            os << center_string(std::to_string(x), max_len[i]);
         }
         if (type == data_type::int32) {
            int32_t x = std::get<int32_t>(df._node->at(name, j));
            os << center_string(std::to_string(x), max_len[i]);
         }
         if (type == data_type::int64) {
            int64_t x = std::get<int64_t>(df._node->at(name, j));
            os << center_string(std::to_string(x), max_len[i]);
         }
         if (type == data_type::uint32) {
            uint32_t x = std::get<uint32_t>(df._node->at(name, j));
            os << center_string(std::to_string(x), max_len[i]);
         }
         if (type == data_type::uint64) {
            uint64_t x = std::get<uint64_t>(df._node->at(name, j));
            os << center_string(std::to_string(x), max_len[i]);
         }
         if (type == data_type::float32) {
            float x = std::get<float>(df._node->at(name, j));
            os << center_string(std::to_string(x), max_len[i]);
         }
         if (type == data_type::float64) {
            double x = std::get<double>(df._node->at(name, j));
            os << center_string(std::to_string(x), max_len[i]);
         }
         if (type == data_type::string) {
            std::string x = std::get<std::string>(df._node->at(name, j));
            os << center_string(x, max_len[i]);
         }
         os << "|";
      }
      os << "\n";
   }
   // Print +----+
   os << "+";
   for (size_t i = 0; i < cols; i++) {
      for (size_t j = 0; j < max_len[i]; j++) {
         os << "-";
      }
      if (i + 1 < cols) {
         os << "+";
      }
   }
   os << "+\n";

   return os;
}

// Data frame
class dataframe {
 public:
   using vector_type = std::vector<cell_type>;

 private:
   std::shared_ptr<dataframe_node> _node = nullptr;

 public:
   dataframe() {}

   ~dataframe() {}

   // Return whether the data frame is empty
   bool empty() const { return !_node; }

   // Add a column named name from ten::vector
   template <typename T>
   void add_col(const std::string &name, const ten::vector<T> &v) {
      // If the node is empty, create it
      if (!_node) {
         _node = std::make_shared<dataframe_node>(dataframe_node());
      }
      [[unlikely]] if (_node->has_column_name(name)) { return; }
      size_t size = v.size();
      // Set the rows size
      if (_node->_rows == 0) {
         _node->_rows = size;
         // Fill the indices with 0..rows-1
         _node->_indices.resize(size);
         for (size_t i = 0; i < _node->_rows; i++) {
            _node->_indices[i] = i;
         }
      } else if (_node->_rows != size) {
         std::cout << "tenseur dataframe: Vector of different size"
                   << std::endl;
         return;
      }
      vector_type data({size});
      // Copy data into std::vector<cell_type>
      for (size_t i = 0; i < size; i++) {
         data[i] = cell_type(v[i]);
      }
      auto ptr = std::make_shared<vector_type>(data);
      // Set the column name
      _node->_names.push_back(name);
      // Set the type
      if constexpr (std::is_same_v<T, bool>) {
         _node->_types.push_back(data_type::boolean);
      }
      if constexpr (std::is_same_v<T, int32_t>) {
         _node->_types.push_back(data_type::int32);
      }
      if constexpr (std::is_same_v<T, int64_t>) {
         _node->_types.push_back(data_type::int64);
      }
      if constexpr (std::is_same_v<T, uint32_t>) {
         _node->_types.push_back(data_type::uint32);
      }
      if constexpr (std::is_same_v<T, uint64_t>) {
         _node->_types.push_back(data_type::uint64);
      }
      if constexpr (std::is_same_v<T, float>) {
         _node->_types.push_back(data_type::float32);
      }
      if constexpr (std::is_same_v<T, double>) {
         _node->_types.push_back(data_type::float64);
      }
      if constexpr (std::is_same_v<T, std::complex<float>>) {
         _node->_types.push_back(data_type::complexfloat32);
      }
      if constexpr (std::is_same_v<T, std::complex<double>>) {
         _node->_types.push_back(data_type::complexfloat64);
      }
      if constexpr (std::is_same_v<T, std::string>) {
         _node->_types.push_back(data_type::string);
      }
      // Set the data
      _node->_data[name] = ptr;
      // Set the col index
      _node->_cols++;
   }

   // Add a column named name from std::vector
   template <typename T>
   void add_col(const std::string &name, const std::vector<T> &v) {
      // If the node is empty, create it
      if (!_node) {
         _node = std::make_shared<dataframe_node>(dataframe_node());
      }
      [[unlikely]] if (_node->has_column_name(name)) { return; }
      size_t size = v.size();
      // Set the rows size
      if (_node->_rows == 0) {
         _node->_rows = size;
         // Fill the indices with 0..rows-1
         _node->_indices.resize(size);
         for (size_t i = 0; i < _node->_rows; i++) {
            _node->_indices[i] = i;
         }
      } else if (_node->_rows != size) {
         std::cout << "tenseur dataframe: Vector of different size"
                   << std::endl;
         return;
      }
      vector_type data({size});
      // Copy data into std::vector<cell_type>
      for (size_t i = 0; i < size; i++) {
         data[i] = cell_type(v[i]);
      }
      auto ptr = std::make_shared<vector_type>(data);
      // Set the column name
      _node->_names.push_back(name);
      // Set the type
      if constexpr (std::is_same_v<T, bool>) {
         _node->_types.push_back(data_type::boolean);
      }
      if constexpr (std::is_same_v<T, int32_t>) {
         _node->_types.push_back(data_type::int32);
      }
      if constexpr (std::is_same_v<T, int64_t>) {
         _node->_types.push_back(data_type::int64);
      }
      if constexpr (std::is_same_v<T, uint32_t>) {
         _node->_types.push_back(data_type::uint32);
      }
      if constexpr (std::is_same_v<T, uint64_t>) {
         _node->_types.push_back(data_type::uint64);
      }
      if constexpr (std::is_same_v<T, float>) {
         _node->_types.push_back(data_type::float32);
      }
      if constexpr (std::is_same_v<T, double>) {
         _node->_types.push_back(data_type::float64);
      }
      if constexpr (std::is_same_v<T, std::complex<float>>) {
         _node->_types.push_back(data_type::complexfloat32);
      }
      if constexpr (std::is_same_v<T, std::complex<double>>) {
         _node->_types.push_back(data_type::complexfloat64);
      }
      if constexpr (std::is_same_v<T, std::string>) {
         _node->_types.push_back(data_type::string);
      }
      // Set the data
      _node->_data[name] = ptr;
      // Set the col index
      _node->_cols++;
   }

   // Remove a column named name
   void remove(const std::string &name) {
      // Get the index
      long index = _node->index_from_name(name);
      if (index == -1) {
         return;
      }
      _node->remove_column(index);
      _node->remove_types(index);
      _node->remove_data(name);
      _node->_cols--;
      // If there's no more columns left, set the rows to 0
      // and clear the indices
      if (_node->_cols == 0) {
         // Set the rows to 0
         _node->_rows = 0;
         // Clear the indices
         _node->_indices = std::vector<cell_type>();
      }
   }

   // Remove a column from its index
   void remove(const size_t index) {
      std::string name = _node->_names[index];
      if (!_node->has_column_name(name)) {
         return;
      }
      _node->remove_column(index);
      _node->remove_types(index);
      _node->remove_data(name);
      _node->_cols--;
      // If there's no more columns left, set the rows to 0
      // and clear the indices
      if (_node->_cols == 0) {
         // Set the rows to 0
         _node->_rows = 0;
         // Clear the indices
         _node->_indices = std::vector<cell_type>();
      }
   }

   /*
   // Remove a row (return a data frame view)
   void remove_row(const size_t index) {
   }*/

   // Select indices and columns names
   dataframe_view select(const std::vector<size_t> &row_indices,
                         const std::vector<std::string> &names) {
      return dataframe_view(_node, row_indices, names);
   }

   // Select indices and column indices
   dataframe_view select(const std::vector<size_t> &row_indices,
                         const std::vector<size_t> &col_indices) {
      std::vector<std::string> names = _node->column_names(col_indices);
      return dataframe_view(_node, row_indices, names);
   }

   // Select using indices sequences and columns names
   dataframe_view select(const ten::seq &row_seq,
                         const std::vector<std::string> &names) {
      size_t size = row_seq._end - row_seq._start;
      std::vector<size_t> row_indices(size);
      for (size_t i = 0; i < size; i++) {
         row_indices[i] = i + row_seq._start;
      }
      return dataframe_view(_node, row_indices, names);
   }

   // Select using indices sequence and column indices
   dataframe_view select(const ten::seq &row_seq,
                         const std::vector<size_t> &col_indices) {
      size_t size = row_seq._end - row_seq._start;
      std::vector<size_t> row_indices(size);
      for (size_t i = 0; i < size; i++) {
         row_indices[i] = i + row_seq._start;
      }
      std::vector<std::string> names = _node->column_names(col_indices);
      return dataframe_view(_node, row_indices, names);
   }

   // Select columns by names using [] operator
   dataframe_view operator[](const std::vector<std::string> &names) {

      std::vector<size_t> row_indices(_node->_rows);
      for (size_t i = 0; i < _node->_rows; i++) {
         row_indices[i] = i;
      }
      return dataframe_view(_node, row_indices, names);
   }

   // Select columns by indices using [] operator
   dataframe_view operator[](const std::vector<size_t> &col_indices) {
      std::vector<size_t> row_indices(_node->_rows);
      for (size_t i = 0; i < _node->_rows; i++) {
         row_indices[i] = i;
      }
      size_t size = col_indices.size();
      std::vector<std::string> names = _node->column_names(col_indices);
      return dataframe_view(_node, row_indices, names);
   }

   // select columns by indices using ten::seq
   dataframe_view operator[](const ten::seq &col_seq) {
      std::vector<size_t> row_indices(_node->_rows);
      for (size_t i = 0; i < _node->_rows; i++) {
         row_indices[i] = i;
      }
      size_t size = col_seq._end - col_seq._start;
      std::vector<size_t> col_indices({size});
      for (size_t i = 0; i < size; i++) {
         col_indices[i] = col_seq._start + i;
      }
      std::vector<std::string> names = _node->column_names(col_indices);
      return dataframe_view(_node, row_indices, names);
   }

   friend std::ostream &operator<<(std::ostream &os, const dataframe &df);
};

std::ostream &operator<<(std::ostream &os, const dataframe &df) {
   os << "dataframe[" << df._node->_rows << "x" << df._node->_cols << "]\n";
   size_t cols = df._node->_cols;
   size_t rows = df._node->_rows;
   std::vector<size_t> max_len({cols});
   for (size_t i = 0; i < cols; i++) {
      std::string name = df._node->_names[i];
      max_len[i] = name.size();
      data_type type = df._node->_types[i];
      for (size_t j = 0; j < rows; j++) {
         if (type == data_type::boolean) {
            bool x = std::get<bool>(df._node->at(name, j));
            size_t len = std::to_string(x).size();
            max_len[i] = std::max(max_len[i], len);
         }
         if (type == data_type::int32) {
            int32_t x = std::get<int32_t>(df._node->at(name, j));
            size_t len = std::to_string(x).size();
            max_len[i] = std::max(max_len[i], len);
         }
         if (type == data_type::int64) {
            int64_t x = std::get<int64_t>(df._node->at(name, j));
            size_t len = std::to_string(x).size();
            max_len[i] = std::max(max_len[i], len);
         }
         if (type == data_type::uint32) {
            uint32_t x = std::get<uint32_t>(df._node->at(name, j));
            size_t len = std::to_string(x).size();
            max_len[i] = std::max(max_len[i], len);
         }
         if (type == data_type::uint64) {
            uint64_t x = std::get<uint64_t>(df._node->at(name, j));
            size_t len = std::to_string(x).size();
            max_len[i] = std::max(max_len[i], len);
         }
         if (type == data_type::float32) {
            float x = std::get<float>(df._node->at(name, j));
            size_t len = std::to_string(x).size();
            max_len[i] = std::max(max_len[i], len);
         }
         if (type == data_type::float64) {
            double x = std::get<double>(df._node->at(name, j));
            size_t len = std::to_string(x).size();
            max_len[i] = std::max(max_len[i], len);
         }
         if (type == data_type::string) {
            std::string x = std::get<std::string>(df._node->at(name, j));
            size_t len = x.size();
            max_len[i] = std::max(max_len[i], len);
         }
      }
   }
   // Print +----+
   os << "+";
   for (size_t i = 0; i < cols; i++) {
      for (size_t j = 0; j < max_len[i]; j++) {
         os << "-";
      }
      if (i + 1 < cols) {
         os << "+";
      }
   }
   os << "+\n";
   os << "|";
   for (size_t i = 0; i < cols; i++) {
      std::string name = df._node->_names[i];
      os << center_string(name, max_len[i]);
      os << "|";
   }
   os << "\n";
   // Print +----+
   os << "+";
   for (size_t i = 0; i < cols; i++) {
      for (size_t j = 0; j < max_len[i]; j++) {
         os << "-";
      }
      if (i + 1 < cols) {
         os << "+";
      }
   }
   os << "+\n";
   // Print the values
   for (size_t j = 0; j < rows; j++) {
      os << "|";
      for (size_t i = 0; i < cols; i++) {
         std::string name = df._node->_names[i];
         data_type type = df._node->_types[i];
         if (type == data_type::boolean) {
            bool x = std::get<bool>(df._node->at(name, j));
            os << center_string(std::to_string(x), max_len[i]);
         }
         if (type == data_type::int32) {
            int32_t x = std::get<int32_t>(df._node->at(name, j));
            os << center_string(std::to_string(x), max_len[i]);
         }
         if (type == data_type::int64) {
            int64_t x = std::get<int64_t>(df._node->at(name, j));
            os << center_string(std::to_string(x), max_len[i]);
         }
         if (type == data_type::uint32) {
            uint32_t x = std::get<uint32_t>(df._node->at(name, j));
            os << center_string(std::to_string(x), max_len[i]);
         }
         if (type == data_type::uint64) {
            uint64_t x = std::get<uint64_t>(df._node->at(name, j));
            os << center_string(std::to_string(x), max_len[i]);
         }
         if (type == data_type::float32) {
            float x = std::get<float>(df._node->at(name, j));
            os << center_string(std::to_string(x), max_len[i]);
         }
         if (type == data_type::float64) {
            double x = std::get<double>(df._node->at(name, j));
            os << center_string(std::to_string(x), max_len[i]);
         }
         if (type == data_type::string) {
            std::string x = std::get<std::string>(df._node->at(name, j));
            os << center_string(x, max_len[i]);
         }
         os << "|";
      }
      os << "\n";
   }
   // Print +----+
   os << "+";
   for (size_t i = 0; i < cols; i++) {
      for (size_t j = 0; j < max_len[i]; j++) {
         os << "-";
      }
      if (i + 1 < cols) {
         os << "+";
      }
   }
   os << "+\n";

   return os;
}

} // namespace ten

#endif
