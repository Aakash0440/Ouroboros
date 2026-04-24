from ouroboros.environments.base import ObservationEnvironment

class FibonacciModEnv(ObservationEnvironment):
    def __init__(self, modulus=7, seeds=(0, 1), seed=42):
        self.modulus = modulus
        self.seeds = tuple(seeds)
        self.max_lag = 2
        super().__init__(alphabet_size=modulus, name=f"Fibonacci({modulus})", seed=seed)

    def generate(self, length: int, start: int = 0):
        seq = list(self.seeds)
        while len(seq) < length + start:
            seq.append((seq[-1] + seq[-2]) % self.modulus)
        return seq[start:start + length]

    def _generate_stream(self, length: int = 1000):
        return self.generate(length)