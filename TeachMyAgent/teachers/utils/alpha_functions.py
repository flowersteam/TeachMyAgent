from abc import ABC, abstractmethod
import torch


class AlphaFunction(ABC):

    @abstractmethod
    def __call__(self, iteration, average_reward, kl_divergence):
        pass


class PercentageAlphaFunction(AlphaFunction):

    def __init__(self, offset, percentage):
        self.offset = offset
        self.percentage = percentage

    def __call__(self, iteration, average_reward, kl_divergence):
        if iteration < self.offset:
            alpha = 0.
        else:
            kl_divergence = torch.clamp(kl_divergence, min=1e-10)
            average_reward = 0. if average_reward < 0. else average_reward
            alpha = torch.clamp(self.percentage * average_reward / kl_divergence, max=1e5)
        return alpha
