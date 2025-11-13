from abc import ABC, abstractmethod
from typing import List, Tuple

class Fitness(ABC):
    @abstractmethod
    def fitness(self, member: List[float]) -> float:
        pass

    @abstractmethod
    def fitness_bitstring(self, bitstring: str) -> float:
        pass

    @abstractmethod
    def decode_bitstring(self, bitstring: str) -> Tuple[float, float]:
        pass