""" Script for On-NAS & Two-Fold Meta-learning(TFML) & On-NAS

This code have been written for a research purpose. 

Licenses and code references will be added at camera-ready version of the code. 

"""

from abc import ABC
from collections import namedtuple


Task = namedtuple("Task", ["train_loader", "valid_loader", "test_loader"])


class TaskDistribution(ABC):
    """Base class to sample tasks for meta training"""

    def sample_meta_train(self):
        """Sample a meta batch for training

        Returns:
            A list of tasks
        """
        raise NotImplementedError

    def sample_meta_valid(self):
        """Sample a meta batch for validation

        Returns:
            A list of tasks
        """
        raise NotImplementedError

    def sample_meta_test(self):
        """Sample a meta batch for testing

        Returns:
            A list of tasks
        """
        raise NotImplementedError
