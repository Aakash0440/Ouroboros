from .base import ObservationEnvironment
import random

class ModularArithmeticEnv(ObservationEnvironment):

    def __init__(self, modulus=7):
        self.modulus = modulus

    def _generate_stream(self):
        while True:
            a = random.randint(0, self.modulus - 1)
            b = random.randint(0, self.modulus - 1)
            yield (a, b, (a + b) % self.modulus)