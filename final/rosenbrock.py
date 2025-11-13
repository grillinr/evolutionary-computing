from .fitness import Fitness

class Rosenbrock(Fitness):
    def fitness(self, member):
        x, y = member[0], member[1]
        rosenbrock_value = (1 - x)**2 + 100 * (y - x**2)**2
        return 1.0 / (1.0 + rosenbrock_value)

    def fitness_bitstring(self, bitstring):
        x, y = self.decode_bitstring(bitstring)
        rosenbrock_value = (1 - x)**2 + 100 * (y - x**2)**2
        return 1.0 / (1.0 + rosenbrock_value)

    def decode_bitstring(self, bitstring):
        if len(bitstring) % 2 != 0:
            raise ValueError("Bitstring length must be even for Rosenbrock decoding")
        half = len(bitstring) // 2
        x_bits = bitstring[:half]
        y_bits = bitstring[half:]
        x = int(x_bits, 2)
        y = int(y_bits, 2)
        max_val = 2**len(x_bits) - 1
        x = (x / max_val) * 4.0 - 2.0
        y = (y / max_val) * 4.0 - 2.0
        return x, y