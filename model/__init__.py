import numpy as np
import random


class LSH:
    def __init__(self, num_hashes, num_buckets):
        self.num_hashes = num_hashes
        self.num_buckets = num_buckets
        self.hash_functions = [self._generate_hash_function() for _ in range(num_hashes)]
        self.hash_tables = [{} for _ in range(num_hashes)]

    def _generate_hash_function(self):
        a = random.randint(1, self.num_buckets - 1)
        b = random.randint(0, self.num_buckets - 1)
        return lambda x: (a * x + b) % self.num_buckets

    def _hash(self, values, hash_function):
        hashed_values = [hash_function(v) for v in values]
        return tuple(hashed_values)

    def index(self, values):
        for i in range(self.num_hashes):
            hash_value = self._hash(values, self.hash_functions[i])
            if hash_value not in self.hash_tables[i]:
                self.hash_tables[i][hash_value] = []
            self.hash_tables[i][hash_value].append(tuple(values))

    def query(self, values):
        candidates = set()
        for i in range(self.num_hashes):
            hash_value = self._hash(values, self.hash_functions[i])
            if hash_value in self.hash_tables[i]:
                candidates.update(self.hash_tables[i][hash_value])
        return list(candidates)

    
