from .base import ObservationEnvironment
import math

class MultiScaleEnv(ObservationEnvironment):
    def __init__(self, scales=None, modulus=12):
        self.scales = scales or [1, 4, 16]
        self.modulus = modulus

    def generate(self, length: int):
        return [sum(int(math.sin(t / s) * 3) for s in self.scales) % self.modulus for t in range(length)]

    def _generate_stream(self):
        t = 0
        while True:
            yield sum(int(math.sin(t / s) * 3) for s in self.scales) % self.modulus
            t += 1

    @property
    def name(self):
        return f'MultiScale({self.scales})'
