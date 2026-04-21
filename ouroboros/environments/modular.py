from .base import ObservationEnvironment
import random

class ModularArithmeticEnv(ObservationEnvironment):

    def __init__(self, modulus=7, slope=1, intercept=0):
        self.modulus = modulus
        self.slope = slope
        self.intercept = intercept

    def generate(self, length: int):
        return [(self.slope * t + self.intercept) % self.modulus for t in range(length)]

    def _generate_stream(self):
        while True:
            a = random.randint(0, self.modulus - 1)
            b = random.randint(0, self.modulus - 1)
            yield (a, b, (a + b) % self.modulus)
