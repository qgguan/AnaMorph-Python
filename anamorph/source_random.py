# Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).
# Recommended citations retained from the original source:
# Mörschel K, Breit M, Queisser G. Generating neuron geometries for detailed
# three-dimensional simulations using AnaMorph. Neuroinformatics (2017).
# Grein S, Stepniewski M, Reiter S, Knodel MM, Queisser G. 1D-3D hybrid
# modelling - from multi-compartment models to full resolution models in space
# and time. Frontiers in Neuroinformatics 8, 68 (2014).
# Breit M, Stepniewski M, Grein S, Gottmann P, Reinhardt L, Queisser G.
# Anatomically detailed and large-scale simulations studying synapse loss
# and synchrony using NeuroBox. Frontiers in Neuroanatomy 10 (2016).
# See LICENSE and NOTICE.md for attribution.
"""Optional random sequence compatible with the tested Windows C runtime."""
import random


class WindowsCRandom:
    """Local MSVCRT-style ``srand``/``rand`` sequence without global state.

    C's rand algorithm is implementation-defined; this reproduces the local
    Windows reference build, not arbitrary C++ platforms.
    """
    def __init__(self, seed):
        if type(seed) is not int:
            raise ValueError("random seed must be an integer")
        self.state = seed & 0xffffffff

    def rand(self):
        self.state = (214013*self.state + 2531011) & 0xffffffff
        return (self.state >> 16) & 0x7fff

    def uniform(self, low, high):
        return low + (self.rand()/32767.)*(high-low)


def make_random(seed, mode):
    if type(seed) is not int:
        raise ValueError("random seed must be an integer")
    if mode == "python":
        return random.Random(seed)
    if mode == "windows-c":
        return WindowsCRandom(seed)
    raise ValueError("random generator must be 'python' or 'windows-c'")
