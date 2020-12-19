from .base import AbstractNegativeSampler

from tqdm import trange

import numpy as np


class RandomNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random'

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        negative_samples = {}
        print('Sampling negative items')
        for user in trange(1, self.user_count + 1):
            # Get all items that have been seen by the user
            if isinstance(self.train[user][1], tuple): # If the item is a tuple
                seen = set(x[0] for x in self.train[user])
                seen.update(x[0] for x in self.val[user])
                seen.update(x[0] for x in self.test[user])
            else:
                seen = set(self.train[user])
                seen.update(self.val[user])
                seen.update(self.test[user])

            samples = []
            for _ in range(self.sample_size):
                item = np.random.choice(self.item_count) + 1
                # If the item has already been seen or is in negative sample
                while item in seen or item in samples:
                    item = np.random.choice(self.item_count) + 1
                samples.append(item)

            negative_samples[user] = samples

        return negative_samples
