#ifndef TENSEUR_NUMBER_THEORY_SIEVE
#define TENSEUR_NUMBER_THEORY_SIEVE

#include <ten/tensor.hxx>
#include <cmath>

namespace ten{
namespace number_theory{

// Return all prime numbers <= n
template<class __t = std::size_t>
auto sieve(std::size_t n) {
   // Primes numbers from 2 to n (n-1)
   std::vector<bool> primes(n-1, true);
   size_t sqrtn = std::floor(std::sqrt(n));
   for (size_t k = 2; k <= sqrtn; k++) {
      if (primes[k-2]) {
         for (size_t l = k*k; l <= n; l += k) {
            primes[l - 2] = false;
         }
      }
   }
   size_t number_primes = 0;
   for (size_t k = 0; k < n-1; k++) {
      number_primes += primes[k];
   }
   ten::vector<std::size_t> numbers(number_primes);
   size_t i = 0;
   for (size_t k = 2; k <= n; k++) {
      if (primes[k-2]) {
            numbers[i] = k;
            i++;
      }
   }
   return numbers;
}


}
}


#endif
