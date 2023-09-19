from typing import TypeVar, Generic, List, Any, Optional, Mapping

T = TypeVar('T')
N = TypeVar('N')
#N = Optional[Any]

print(T)

class Tensor(Generic[T]):
  def __init__(self):
    self.shape: List[T] = []

t = Tensor[float]()

print(t)

