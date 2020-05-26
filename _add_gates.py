""" Useful additional gates as projectQ Gate subclasses
"""
from projectq.ops import SelfInverseGate
import numpy as np

# two-qubit same-Pauli gates
class XXGate(SelfInverseGate):
    """ Pauli-XX gate class """
    def __str__(self):
        return "XX"

    @property
    def matrix(self):
        return np.matrix([[0, 0, 0, 1],
                          [0, 0, 1, 0],
                          [0, 1, 0, 0],
                          [1, 0, 0, 0]])

XX = XXGate()

class YYGate(SelfInverseGate):
    """ Pauli-YY gate class """
    def __str__(self):
        return "YY"

    @property
    def matrix(self):
        return np.matrix([[0,  0, 0, -1],
                          [0,  0, 1,  0],
                          [0,  1, 0,  0],
                          [-1, 0, 0,  0]])

YY = YYGate()

class ZZGate(SelfInverseGate):
    """ Pauli-ZZ gate class """
    def __str__(self):
        return "ZZ"

    @property
    def matrix(self):
        return np.matrix([[1,  0,  0, 0],
                          [0, -1,  0, 0],
                          [0,  0, -1, 0],
                          [0,  0,  0, 1]])

ZZ = ZZGate()
