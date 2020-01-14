import torch
from collections import namedtuple, deque
import random
import numpy as np

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, num_agents, cuda=False):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): Maximum size of buffer
            num_agents (int): The number of agents sharing this buffer.
            cuda (bool): Whether to send sampled experiences to the GPU
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.n_agents = num_agents
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        if cuda:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
    
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    
    def sample(self, batch_size):
        """Randomly sample batch_size experiences from memory.
        
        Returns a tuple (states, actions, rewards, next_states, dones).
        Each tuple element is a list of tensors, where tensor i corresponds to the ith agent.
        This makes concatenation and extracting the individual experience of the ith agent easy.
        The jth row in each tensor corresponds to the jth sampled experience.
        """
        # Convenience functions
        def torchify(some_list):
            return torch.from_numpy(np.vstack(some_list)).float().to(self.device)
        def torchify_uint8(some_list):
            return torch.from_numpy(np.vstack(some_list).astype(np.uint8)).float().to(self.device)
        # Actual sampling    
        experiences = random.sample(self.memory, k=batch_size)
        # Convert to list of one tensor per agent (can be looped over easily)
        states      = [torchify([e.state[n] for e in experiences if e is not None]) for n in range(self.n_agents)]
        next_states = [torchify([e.next_state[n] for e in experiences if e is not None]) for n in range(self.n_agents)]
        actions     = [torchify([e.action[n] for e in experiences if e is not None]) for n in range(self.n_agents)]
        rewards     = [torchify([e.reward[n] for e in experiences if e is not None]) for n in range(self.n_agents)]
        dones       = [torchify_uint8([e.done[n] for e in experiences if e is not None]) for n in range(self.n_agents)]
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)