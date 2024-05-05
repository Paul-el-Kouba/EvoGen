import math
import numpy as np


class TS_SS:
    def __init__(self):
        pass

    def cosine(self, vec1, vec2):
        vec1 = np.array(vec1, dtype=float)
        vec2 = np.array(vec2, dtype=float)

        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0  # Avoid division by zero if one of the vectors is zero
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)

        return cosine_similarity

    def euclidean(self, vec1, vec2):
        return math.sqrt(sum(math.pow((v1 - v2), 2) for v1, v2 in zip(vec1, vec2)))

    def theta(self, vec1, vec2):
        x = self.cosine(vec1, vec2)
        if x < 0:
            x = 0
        elif x > 1:
            x = 1
        return math.acos(x) + math.radians(10)

    def triangle(self, vec1, vec2):
        theta = math.radians(self.theta(vec1, vec2))
        return (np.linalg.norm(vec1) * np.linalg.norm(vec2) * math.sin(theta)) / 2

    def magnitude_difference(self, vec1, vec2):
        return abs(np.linalg.norm(vec1) - np.linalg.norm(vec2))

    def sector(self, vec1, vec2):
        ED = self.euclidean(vec1, vec2)
        MD = self.magnitude_difference(vec1, vec2)
        theta = self.theta(vec1, vec2)
        return math.pi * math.pow((ED + MD), 2) * theta / 360

    def TS_SS(self, vec1, vec2):
        return 1 / (1 + self.triangle(vec1, vec2) * self.sector(vec1, vec2))
