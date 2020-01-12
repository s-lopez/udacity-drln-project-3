import numpy as np
import copy

class OUNoise:
    """Ornstein-Uhlenbeck noise process."""

    def __init__(self, size, mu=0., theta=1, sigma=.15, decay_rate=.95):
        """Initialize parameters and noise process."""
        self.mu = np.array([mu for i in range(size)])
        self.theta = theta
        self.sigma = sigma
        self.decay = 1
        self.decay_rate = decay_rate
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.random(len(x))
        self.state = x + dx
        return self.state * self.decay
        #return torch.tensor(self.state * self.scale).float()
        
    def decay_step(self):
        self.decay *= self.decay_rate
    
    