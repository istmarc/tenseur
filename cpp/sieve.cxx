#include <ten/tensor>
#include <ten/number_theory/sieve.hxx>


int main() {
   auto primes = ten::number_theory::sieve(100);
   for (size_t i = 0; i < primes.size(); i++) {
      std::cout << primes[i] << std::endl;
   }

}

