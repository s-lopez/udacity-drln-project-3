import torch
from collections import namedtuple, deque
import random
import numpy as np

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, cuda=False):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        if cuda:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, batch_size, n_agents):
        """Randomly sample a batch of experiences from memory."""
        def torchify(some_list):
            return torch.from_numpy(np.vstack(some_list)).float().to(self.device)
        def torchify_uint8(some_list):
            return torch.from_numpy(np.vstack(some_list).astype(np.uint8)).float().to(self.device)
            
        experiences = random.sample(self.memory, k=batch_size)
        # List of one states-tensor per agent for all agents (can be looped over easily)
        states      = [torchify([e.state[n] for e in experiences if e is not None]) for n in range(n_agents)]
        next_states = [torchify([e.next_state[n] for e in experiences if e is not None]) for n in range(n_agents)]
        actions     = [torchify([e.action[n] for e in experiences if e is not None]) for n in range(n_agents)]
        rewards     = [torchify([e.reward[n] for e in experiences if e is not None]) for n in range(n_agents)]
        dones       = [torchify_uint8([e.done[n] for e in experiences if e is not None]) for n in range(n_agents)]
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)