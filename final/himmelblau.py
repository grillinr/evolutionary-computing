from .fitness import Fitness

class Himmelblau(Fitness):
    def fitness(self, member):
        x, y = member[0], member[1]
        himmelblau_value = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
        return 1.0 / (1.0 + himmelblau_value)

    def fitness_bitstring(self, bitstring):
        x, y = self.decode_bitstring(bitstring)
        himmelblau_value = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
        return 1.0 / (1.0 + himmelblau_value)

    def decode_bitstring(self, bitstring):
        if len(bitstring) % 2 != 0:
            raise ValueError("Bitstring length must be even for Himmelblau decoding")
        half = len(bitstring) // 2
        x_bits = bitstring[:half]
        y_bits = bitstring[half:]
        x = int(x_bits, 2)
        y = int(y_bits, 2)
        max_val = 2**len(x_bits) - 1
        x = (x / max_val) * 20.0 - 10.0
        y = (y / max_val) * 20.0 - 10.0
        return x, y