import numpy as np
from typing import List
from similarity_measures import TS_SS


class KNN(object):
    def __init__(self, k=3):
        self.x_train = None
        self.y_train = None
        self.k = k

    def fit(self, x, y) -> None:
        """
        Fit the model to the data

        For K-Nearest neighbors, the model is the data, so we just
        need to store the data

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels
        """
        self.x_train = x
        self.y_train = y

    def predict(self, x) -> List:
        """
        Predict x from the k-nearest neighbors

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            A vector of size N of the predicted class for each sample in x
        """

        sim = TS_SS()
        ret = []
        for i in self.x_train:
            ret.append(sim.TS_SS(i, x))

        ret = [self.y_train[x] for x in np.argsort(-np.asarray(ret))][-25:]

        return ret
