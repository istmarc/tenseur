#include <ten/tensor>
#include <ten/number_theory/sieve.hxx>

auto naive(size_t n) {
   std::vector<bool> primes(n-1, true);
   for (size_t k = 2; k <=n; k++) {
      for (size_t l = 2; l < k; l++) {
         if (k % l == 0) {
	    primes[k - 2] = false;
	    break;
	 }
      }
   }
   size_t nprimes = 0;
   for (size_t k = 0; k < n-1; k++) {
      nprimes += primes[k]; 
   }
   ten::vector<size_t> numbers(nprimes);
   size_t i = 0;
   for (size_t k = 2; k <= n; k++) {
      if (primes[k-2]){
         numbers[i] = k;
	 i++;
      }
   }
   return numbers;
}

int main() {
   auto real_primes = naive(100000);
   size_t nrealprimes = real_primes.size();
   std::cout << "Naive found " << nrealprimes << " primes" << std::endl;
   auto primes = ten::number_theory::sieve(100000);
   size_t nprimes = primes.size();
   std::cout << "Sieve found " << nprimes << " primes" << std::endl;

   size_t s = 0;
   for (size_t i = 0; i < nprimes; i++) {
      s += (real_primes[i] != primes[i]);
   }
   if (s > 0) {
      std::cout << "Different" << std::endl;	
   } else {
      std::cout << "Same" << std::endl;
   }

}

