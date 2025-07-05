#ifndef TENSEUR_NUMBER_THEORY_SIEVE
#define TENSEUR_NUMBER_THEORY_SIEVE

#include <ten/tensor.hxx>
#include <cmath>

namespace ten{
namespace ntheory{

// Return all prime numbers <= n
template<class T = long int>
auto sieve(long int n) {
   // Primes numbers from 2 to n (n-1)
   std::vector<bool> primes(n-1, true);
   long int sqrtn = std::floor(std::sqrt(n));
   for (long int k = 2; k <= sqrtn; k++) {
      if (primes[k-2]) {
         for (long int l = k*k; l <= n; l += k) {
            primes[l - 2] = false;
         }
      }
   }
   long int number_primes = 0;
   for (long int k = 0; k < n-1; k++) {
      number_primes += primes[k];
   }
   ten::vector<long int> numbers(number_primes);
   long int i = 0;
   for (long int k = 2; k <= n; k++) {
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
