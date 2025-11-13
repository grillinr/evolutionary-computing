from .fitness import Fitness

class MaxOnes(Fitness):
    def fitness(self, member):
        return 0.0

    def fitness_bitstring(self, bitstring):
        ones = sum(1 for c in bitstring if c == '1')
        return ones / len(bitstring)

    def decode_bitstring(self, bitstring):
        return 0.0, 0.0