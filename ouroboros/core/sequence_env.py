class SequenceEnvironment:
    """
    Fully compatible environment wrapper for OUROBOROS pipeline
    """

    def __init__(self, sequence, alphabet_size=None):
        self.sequence = sequence
        self.t = 0

        # Required by system
        self.alphabet_size = alphabet_size if alphabet_size is not None else len(set(sequence))
        self.max_steps = len(sequence)

        # Optional but often expected
        self.observation_space = list(range(self.alphabet_size))
        self.action_space = list(range(self.alphabet_size))

    def reset(self, stream_length=None):
        """
        Reset environment with optional stream length
        """
        self.t = 0

        if stream_length is not None:
            self.max_steps = min(stream_length, len(self.sequence))
        else:
            self.max_steps = len(self.sequence)

        return self.sequence[0]

    def step(self, action=None):
        self.t += 1

        if self.t >= self.max_steps:
            done = True
            obs = self.sequence[self.max_steps - 1]
        else:
            done = False
            obs = self.sequence[self.t]

        reward = 0.0
        return obs, reward, done, {}

    def peek_all(self):
        """
        REQUIRED: return full observable stream
        """
        return self.sequence[:self.max_steps]