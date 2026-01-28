from cudaq_qlsa.qlsa.base import QLSA
from cudaq_qlsa.qlsa.hhl_dependencies import QFT, invQFT, A_exp_pauli, NA_exp_pauli
from cudaq_qlsa.qlsa.PauliDecompose import PauliDecomposition
from typing import Optional
import cudaq
import warnings
import numpy as np
import math

class HHL(QLSA):
    def __init__(
        self,
        readout: str,
        num_qpe_qubits: int,
        t0: float,
        swap_test_vector: Optional[np.ndarray] = None
    ):
        """
        Initialize the HHL QLSA.
        Args:
            A: The matrix representing the linear system.
            b: The vector representing the right-hand side of the linear system.
            readout: The readout method to use. Should be either 'measure_x' or 'swap_test'.
            num_qpe_qubits: The number of qubits to use for the QPE.
            t0: The time parameter used in the controlled-Hamiltonian operations.
            swap_test_vector: The vector to use for the swap test. Only used if readout is 'swap_test'.
        """
        
        super().__init__()
        self.readout = readout
        self.num_qpe_qubits = num_qpe_qubits
        self.t0 = t0
        self.swap_test_vector = swap_test_vector

    
    def build_circuit(self, A: np.ndarray, b: np.ndarray):
        """
        Compose the HHL circuit out of the state preparation circuit, the QLSA, and the readout circuit.
        Either calls measure_x_circuit or swap_test_circuit, or estimate_resource depending on the readout method.

        Returns:
            QuantumCircuit: The composed HHL circuit.
        """
        # Check if readout method is valid
        if self.readout not in ("measure_x", "swap_test", "resource_estimate"):
            raise ValueError("readout must be either 'measure_x', 'swap_test', or 'resource_estimate'")
        if self.readout == "swap_test" and self.swap_test_vector is None:
            raise ValueError("swap_test requires `swap_test_vector`.")
        if self.readout == "measure_x" and self.swap_test_vector is not None:
            warnings.warn("swap_test_vector provided but readout is 'measure_x'; ignoring.")
        if self.readout == "resource_estimate" and self.swap_test_vector is not None:
            warnings.warn("swap_test_vector provided but readout is 'resource_estimate'; ignoring.")
        if self.readout == "resource_estimate":
            warnings.warn("Make sure cudaq version is 0.13.0 or newer. cudaq.estimate_resources() is not supported in older versions.")
            
        # Check if A is a square matrix and b is a vector of matching size
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be a square matrix, got shape {A.shape}")
        if b.ndim != 1:
            raise ValueError(f"b must be a vector, got shape {b.shape}")
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"Dimension mismatch: A is {A.shape[0]}x{A.shape[1]}, but b has shape {b.shape}")
        
        # Check if A is Hermitian
        if not np.allclose(A, A.T.conjugate()):
            raise ValueError("A must be Hermitian")
        
        # Check if b has unit norm
        if not np.isclose(np.linalg.norm(b), 1):
            raise ValueError(f"b should have unit norm, instead has norm: {np.linalg.norm(b)}")
        
        # Perform Pauli Decomposition on A
        [words,coefficients] = PauliDecomposition(A)
        words = words[1:]
        coefficients = coefficients[1:]
        coefficients = [float(np.real(x)) for x in coefficients]  

        if self.readout == "measure_x":
            # return self.measure_x_circuit(b, self.num_qpe_qubits, self.t0, coefficients, words)
            return (self.measure_x_circuit, (b.tolist(), self.num_qpe_qubits, self.t0, coefficients, words))
            
        if self.readout == "resource_estimate":
            b_num = int(math.log2(len(b)))
            return (self.resource_estimate_circuit, (b_num, self.num_qpe_qubits, self.t0, coefficients, words))

        elif self.readout == "swap_test":
            return (self.swap_test_circuit, (b.tolist(), self.swap_test_vector, self.num_qpe_qubits, self.t0, coefficients, words))
            
        else:
            raise ValueError(f"Invalid readout method: {self.readout}. Valid methods: measure_x, resource_estimate, swap_test.")
        
    def draw(self, A, b):
        """
        Draw the built circuit
        """
        [words,coefficients] = PauliDecomposition(A)
        words = words[1:]
        coefficients = coefficients[1:]
        coefficients = [float(np.real(x)) for x in coefficients]  
        if self.readout == "measure_x":
            print(cudaq.draw(self.measure_x_circuit, b, self.num_qpe_qubits, self.t0, coefficients, words))
        
        if self.readout == "resource_estimate":
            b_num = int(math.log2(len(b)))
            print(cudaq.draw(self.resource_estimate_circuit, b_num, self.num_qpe_qubits, self.t0, coefficients, words))

        elif self.readout == "swap_test":
            print(cudaq.draw(self.swap_test_circuit, b, self.swap_test_vector, self.num_qpe_qubits, self.t0, coefficients, words))

    @cudaq.kernel
    def measure_x_circuit(b: list[float], num_qpe_qubits: int, t0:float, coefficients: list[float], words: list[cudaq.pauli_word]):
        """
        Build the circuit for measuring the x register.
        """
        #==========================================================================
        # Quantum Circuit
        #==========================================================================
        # Qubits
        qAnc = cudaq.qubit()
        qReg = cudaq.qvector(num_qpe_qubits)
        bReg = cudaq.qvector(b)
        numQubits = len(qReg) + len(bReg) + 1
        
        # Apply Hadamard on register C
        h(qReg)

        # Apply Hamiltonian 
        for i in range(len(qReg)):
            Time = t0/(2**(len(qReg)-1-i))
            cudaq.control(A_exp_pauli, qReg[i], bReg, coefficients, words, Time)
            
        # Apply inverse QFT (QFT of CudaQ is the inverse of QFT of Qiskit)
        QFT(qReg)

        # Apply y rotations on Ancilla qubit 
        for i in range(len(qReg)):
            # ry_angle = (2*np.pi)/(2**(i+1))
            ry.ctrl((2*np.pi)/(2**(i+1)), qReg[i], qAnc)  
            
        #================ Uncompute the circuit ================
        # Apply QFT
        invQFT(qReg)

        # Apply Hamiltonian 
        for i in range(len(qReg)-1,-1,-1):
            Time = t0/(2**(len(qReg) -1 -i))
            cudaq.control(NA_exp_pauli, qReg[i], bReg, coefficients, words, Time)

        # Apply Hadamard on register C
        h(qReg)

        # Measurement
        mz(qAnc)
        mz(bReg)

    @cudaq.kernel
    def resource_estimate_circuit(b_num:int, num_qpe_qubits: int, t0:float, coefficients: list[float], words: list[cudaq.pauli_word]):
        """
        Build the circuit for estimating the resources.
        """
        #==========================================================================
        # Quantum Circuit
        #==========================================================================
        # Qubits
        qAnc = cudaq.qubit()
        qReg = cudaq.qvector(num_qpe_qubits)
        bReg = cudaq.qvector(b_num)
        numQubits = len(qReg) + len(bReg) + 1
        
        # Apply Hadamard on register C
        h(qReg)

        # Apply Hamiltonian 
        for i in range(len(qReg)):
            Time = t0/(2**(len(qReg)-1-i))
            cudaq.control(A_exp_pauli, qReg[i], bReg, coefficients, words, Time)
            
        # Apply inverse QFT (QFT of CudaQ is the inverse of QFT of Qiskit)
        QFT(qReg)

        # Apply y rotations on Ancilla qubit 
        for i in range(len(qReg)):
            # ry_angle = (2*np.pi)/(2**(i+1))
            ry.ctrl((2*np.pi)/(2**(i+1)), qReg[i], qAnc)  
            
        #================ Uncompute the circuit ================
        # Apply QFT
        invQFT(qReg)

        # Apply Hamiltonian 
        for i in range(len(qReg)-1,-1,-1):
            Time = t0/(2**(len(qReg) -1 -i))
            cudaq.control(NA_exp_pauli, qReg[i], bReg, coefficients, words, Time)

        # Apply Hadamard on register C
        h(qReg)

        # Measurement
        mz(qAnc)
        mz(bReg)
        
    @cudaq.kernel
    def swap_test_circuit(b: list[float], 
                          swap_test_vector:list[float], 
                          num_qpe_qubits: int, 
                          t0:float, 
                          coefficients: list[float], 
                          words: list[cudaq.pauli_word]):
        """
        Build the circuit for the swap test. Estimates the inner product of x and v.
        """
        #==========================================================================
        # Quantum Circuit
        #==========================================================================
        # Qubits
        qAnc = cudaq.qubit()
        qReg = cudaq.qvector(num_qpe_qubits)
        bReg = cudaq.qvector(b)
        numQubits = len(qReg) + len(bReg) + 1

        swap_test_qAnc = cudaq.qubit()
        
        
        # Apply Hadamard on register C
        h(qReg)

        # Apply Hamiltonian 
        for i in range(len(qReg)):
            Time = t0/(2**(len(qReg)-1-i))
            cudaq.control(A_exp_pauli, qReg[i], bReg, coefficients, words, Time)
            
        # Apply inverse QFT (QFT of CudaQ is the inverse of QFT of Qiskit)
        QFT(qReg)

        # Apply y rotations on Ancilla qubit 
        for i in range(len(qReg)):
            # ry_angle = (2*np.pi)/(2**(i+1))
            ry.ctrl((2*np.pi)/(2**(i+1)), qReg[i], qAnc)  
            
        #================ Uncompute the circuit ================
        # Apply QFT
        invQFT(qReg)

        # Apply Hamiltonian 
        for i in range(len(qReg)-1,-1,-1):
            Time = t0/(2**(len(qReg) -1 -i))
            cudaq.control(NA_exp_pauli, qReg[i], bReg, coefficients, words, Time)

        # Apply Hadamard on register C
        h(qReg)

        vReg = cudaq.qvector(swap_test_vector) 
        
        # Measurement
        h(swap_test_qAnc)
        for i in range(len(bReg)):
            swap.ctrl(swap_test_qAnc, bReg[i], vReg[i])
        h(swap_test_qAnc)

        mz(qAnc)
        mz(swap_test_qAnc)
