#include <iostream>
#include <ten/tensor>
#include <ten/ntheory>

int main(){
   auto primes = ten::ntheory::sieve(1000);
   std::cout << primes << std::endl;
}
