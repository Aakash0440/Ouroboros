from .base import ObservationEnvironment
import random

class NoiseEnv:
    def __init__(self, modulus=12, alphabet_size=12, seed=42):
        self.modulus = modulus or alphabet_size
        self.seed = seed

    def generate(self, length: int):
        rng = random.Random(self.seed)
        return [rng.randint(0, self.modulus - 1) for _ in range(length)]

    def _generate_stream(self):
        rng = random.Random(self.seed)
        while True:
            yield rng.randint(0, self.modulus - 1)

    @property
    def name(self):
        return f'Noise({self.modulus})'
