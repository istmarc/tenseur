#include <ten/tensor>

#include <any>
#include <unordered_map>

namespace ten::nn {

struct net {
   std::unordered_map<std::string, std::any> _params;

   /// Add a parameter to the network
   void add_param(const std::string &name, std::any parameter) {
      _params[name] = parameter;
   }

   auto params() { return _params; }

   net() {}

   virtual ~net() {}
};

} // namespace ten::nn
