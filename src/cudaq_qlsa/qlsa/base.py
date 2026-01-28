from abc import ABC, abstractmethod
import numpy as np

class QLSA(ABC):

    @abstractmethod
    def build_circuit(self, A: np.ndarray, b: np.ndarray):
        raise NotImplementedError