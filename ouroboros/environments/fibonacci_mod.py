from ouroboros.environments.base import ObservationEnvironment

class FibonacciModEnv:
    def __init__(self, modulus=7, seeds=(0, 1)):
        self.modulus = modulus
        self.seeds = tuple(seeds)
        self.max_lag = 2

    def generate(self, length: int):
        seq = list(self.seeds)
        while len(seq) < length:
            seq.append((seq[-1] + seq[-2]) % self.modulus)
        return seq[:length]

    def _generate_stream(self):
        a, b = self.seeds
        while True:
            yield a
            a, b = b, (a + b) % self.modulus

    @property
    def name(self): return f"Fibonacci({self.modulus})"