import math
from ouroboros.multivariate.beam import MultivariateBeam

n = 80
pos = [int(round(10 * math.cos(0.3 * t))) for t in range(n)]
vel = [int(round(-3 * math.sin(0.3 * t))) for t in range(n)]
acc = [int(round(-0.9 * math.cos(0.3 * t))) for t in range(n)]

channels = {'pos': pos, 'vel': vel, 'acc': acc}
mb = MultivariateBeam(beam_width=25, n_iterations=15)
result = mb.search(channels, target='acc', verbose=True)
print(result.description())