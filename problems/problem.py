import abc

import numpy as np


class Problem(object):
    def __init__(self, sites, h, k, config):
        self._sites = sites
        self._k = k
        self._h = h
        self._operator = self._generate_operator()
        self._initial_value = self._generate_initial_value()

    def initial_function(self, x):
        raise NotImplementedError()

    @property
    def operator(self):
        return self._operator

    @abc.abstractmethod
    def _generate_operator(self):
        pass

    def _generate_initial_value(self):
        initial_value = np.zeros_like(self._sites)
        for index in np.ndindex(*initial_value.shape):
            initial_value[index] = self.initial_function(self._sites[index])
        return initial_value

    @property
    def special_config(self):
        return ""
