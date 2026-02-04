from cudaq_qlsa.generator import generate_problem
from cudaq_qlsa.qlsa.base import QLSA
from cudaq_qlsa.qlsa.hhl import HHL
# from cudaq_qlsa.noise_model import NoiseModeler
from cudaq_qlsa.executer import Executer
from cudaq_qlsa.post_processor import Post_Processor
from cudaq_qlsa.solver import QuantumLinearSolver
from cudaq_qlsa.refiner import Refiner
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import warnings
from typing import Any, Optional

class Tester:

    def __init__(
        self,
        problem_size: int,
        problem: Optional[Any] = None,
        qlsa: Optional[QLSA] = None,
        noisemodel: Optional[Any] = None,
        executer: Optional[Executer] = None,
        post_processor: Optional[Post_Processor] = None,
        verbose: bool = True
    ):
        
        super().__init__()
        self.problem_size = problem_size
        self.problem = problem,
        self.qlsa = qlsa
        self.noisemodel = noisemodel
        self.executer = executer
        self.post_processor = post_processor
        self.verbose = verbose


    

    def GPUvsCPU(self, 
                 sample_size: int, 
                 shots: int, 
                 precision: float, 
                 max_iter: int, 
                 plot: bool = True) -> dict:
        
        if self.noisemodel == None:
            print('Noiseless Experiment!')
            targets = ["qpp-cpu", "nvidia"]
        else: 
            print('Noisy Experiment!')
            targets = ["density-matrix-cpu", "nvidia"]

        time_dict = {}
        iter_dict = {}

        for target in targets:
            time_dict[target] = []
            iter_dict[target] = []
            
        for test in range(sample_size):
            ## Generate a problem
            # -> Call HHL
            prob = generate_problem(n=self.problem_size, cond_number=5.0, sparsity=0.5, seed=0)
            A, b = prob["A"], prob["b"]

            A = A / np.linalg.norm(b)
            b = b / np.linalg.norm(b)

            # Create the solver 
            # -> Call QuantumLinearSolver
            for target in targets:
                # Set the Target
                hhl_solver = QuantumLinearSolver(
                    qlsa = self.qlsa,
                    backend = target,
                    shots = shots,
                    noisemodel = self.noisemodel,
                    executer = self.executer,
                    post_processor = self.post_processor,
                    verbose = self.verbose
                    )
                
                print(f"Running Sample {test} on {target}")

                # Run the IR
                # -> Call the refiner
                refiner = Refiner(A = A, b = b, solver = hhl_solver)
                start = time.time()
                result = refiner.refine(precision, max_iter, plot=False, verbose=self.verbose)
                end = time.time()
                elapsed_time = end - start
                
                iter_dict[target].append(result['total_iterations'])
                time_dict[target].append(elapsed_time)  


        # Plot
        if plot:
            # Time comparison
            plt.figure()
            for key, _ in time_dict.items():
                plt.plot(range(sample_size), time_dict[key], label=f'{key} time')
            plt.xlabel('Instance')
            plt.ylabel('Time (seconds)')
            plt.xticks(range(0,len(time_dict['nvidia']),1 if len(time_dict['nvidia']) < 1000 else 1000))
            if self.noisemodel == None:
                plt.title(f'Runtime Comparison on LSE size {self.problem_size} (GPU vs. QPP-CPU)')
            else:
                plt.title(f'Runtime Comparison on LSE size {self.problem_size} (GPU vs. DMCPU)')
            plt.grid(True)
            plt.legend()
            # plt.savefig(f'Time Comparison size {self.problem_size}-GPUvsDMCPU.png')
            plt.show()

            # Iteration Comparison
            plt.figure()
            for key, _ in iter_dict.items():
                plt.scatter(range(sample_size), iter_dict[key], label=f'{key} Iterations')
            plt.xlabel('Instance')
            plt.ylabel('Time (seconds)')
            plt.xticks(range(0,len(iter_dict['nvidia']),1 if len(iter_dict['nvidia']) < 1000 else 1000))
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.title(f'Iteration Comparison on LSE size {self.problem_size} (GPU vs. DMCPU)')
            plt.grid(True)
            plt.legend()
            # plt.savefig(f'Iteration Comparison size {self.problem_size}-GPUvsDMCPU.png')
            plt.show()

        ...

    def IRvsQPE(self, 
                max_qpe_qubit: int, 
                max_IR_iter: int, 
                precision: float, 
                shots: int, 
                backend: str,
                plot: bool = True
                ) -> dict:

        if backend == 'qpp-cpu' and self.noisemodel:
            raise ValueError("qpp-cpu is selected on a noisy simulation. Switch to density-matrix-cpu for a noisy simulation on CPU.")
        if backend == 'density-matrix-cpu' and self.noisemodel == None:
            warnings.warn("density-matrix-cpu is selected on a noiseless simulation. Switch to qpp-cpu for a noiseless simulation on CPU.")
        
        residual_data = np.zeros((max_qpe_qubit, max_IR_iter+1))

        ## Generate a problem
        # -> Call HHL
        if self.problem == None:
            problem = generate_problem(n=self.problem_size, cond_number=5.0, sparsity=0.5, seed=0)
            A, b = problem["A"], problem["b"]
        else:
            A, b = self.problem[0]["A"], self.problem[0]["b"]

        A = A / np.linalg.norm(b)
        b = b / np.linalg.norm(b)

        for n_qpe_qubits in range(1,max_qpe_qubit+1):
            # Build the circuit
            hhl = HHL(
                readout = 'measure_x',
                num_qpe_qubits = n_qpe_qubits,
                t0 = 2 * np.pi)

            # Create the solver 
            # -> Call QuantumLinearSolver
            hhl_solver = QuantumLinearSolver(
                qlsa = hhl,
                backend = backend,
                shots = shots,
                noisemodel = self.noisemodel,
                executer = self.executer,
                post_processor = self.post_processor,
                verbose = self.verbose
                )
                
            # Run the IR
            # -> Call the refiner
            refiner = Refiner(A = A, b = b, solver = hhl_solver)
            result = refiner.refine(precision, max_IR_iter, plot=False, verbose=self.verbose)
            
            residual_data[n_qpe_qubits-1] = result['residuals']
            
        if plot:
            data = residual_data

            plt.figure(figsize=(6,5))

            im = plt.imshow(data, aspect='auto', cmap='viridis')

            plt.xlabel("IR Iteration")
            plt.ylabel("QPE Qubits")
            plt.title(f"{self.problem_size}x{self.problem_size} System")

            plt.xticks(np.arange(data.shape[1]))
            plt.yticks(np.arange(data.shape[0]), labels=np.arange(1, data.shape[0]+1))

            # Reverse y-axis so that 1 is at the bottom
            plt.gca().invert_yaxis()

            cbar = plt.colorbar(im)
            cbar.set_label("Residual")

            plt.savefig(f"{self.problem_size}x{self.problem_size}_QPE")

            plt.tight_layout()
            plt.show()


    def IRvsShots(self, 
                  shots_list: list[int],
                  max_IR_iter_list: list[int], 
                  precision: float, 
                  shots: int, 
                  backend: str,
                  plot: bool = True
                  ) -> dict:

        if backend == 'qpp-cpu' and self.noisemodel:
            raise ValueError("qpp-cpu is selected on a noisy simulation. Switch to density-matrix-cpu for a noisy simulation on CPU.")
        if backend == 'density-matrix-cpu' and self.noisemodel == None:
            warnings.warn("density-matrix-cpu is selected on a noiseless simulation. Switch to qpp-cpu for a noiseless simulation on CPU.")
            
        shots_data = np.zeros((len(shots_list), 1))
        IR_data = {}
        for maxiter in max_IR_iter_list:
            iter_index = max_IR_iter_list.index(maxiter)
            IR_data[iter_index] = np.zeros((len(shots_list), 1))

        ## Generate a problem
        # -> Call HHL
        if self.problem == None:
            problem = generate_problem(n=self.problem_size, cond_number=5.0, sparsity=0.5, seed=0)
            A, b = problem["A"], problem["b"]
        else:
            A, b = self.problem[0]["A"], self.problem[0]["b"]

        A = A / np.linalg.norm(b)
        b = b / np.linalg.norm(b)
  
        if len(max_IR_iter_list) != 1:
            shots_data = np.zeros((len(shots_list), 1))
            IR_data = {}
            for maxiter in max_IR_iter_list:
                iter_index = max_IR_iter_list.index(maxiter)
                IR_data[iter_index] = np.zeros((len(shots_list), 1))
            
            for shots in shots_list:
                # Create the solver 
                # -> Call QuantumLinearSolver
                hhl_solver = QuantumLinearSolver(
                    qlsa = self.qlsa,
                    backend = backend,
                    shots = shots,
                    noisemodel = self.noisemodel,
                    executer = self.executer,
                    post_processor = self.post_processor,
                    verbose = self.verbose
                    )
                
                index = shots_list.index(shots)
                # Single Run
                refiner = Refiner(A = A, b = b, solver = hhl_solver)
                result = refiner.refine(precision, 0, plot=False, verbose=False)
                shots_data[index] = result['residuals']
                
                # IR Run with variable max iterations
                for maxiter in max_IR_iter_list:
                    iter_index = max_IR_iter_list.index(maxiter)

                    refiner = Refiner(A = A, b = b, solver = hhl_solver)
                    result = refiner.refine(precision, maxiter, plot=False, verbose=self.verbose)

                    r_IR = result['residuals']
                    IR_data[iter_index][index] = r_IR[-1]

        else:
            shots_data = np.zeros((len(shots_list), max_IR_iter_list[0]+1))   
            for shots in shots_list:    
                index = shots_list.index(shots)
                # Create the solver 
                # -> Call QuantumLinearSolver
                hhl_solver = QuantumLinearSolver(
                    qlsa = self.qlsa,
                    backend = backend,
                    shots = shots,
                    noisemodel = self.noisemodel,
                    executer = self.executer,
                    post_processor = self.post_processor,
                    verbose = self.verbose
                    )
                
                # Run the IR
                # -> Call the refiner
                refiner = Refiner(A = A, b = b, solver = hhl_solver)
                result = refiner.refine(precision, max_IR_iter_list[0], plot=False, verbose=self.verbose)
                shots_data[index] = result['residuals']
            
        if plot:
            if len(max_IR_iter_list) != 1:
                plt.figure(figsize=(8, 5))
                plt.plot(range(len(shots_data)), shots_data, linestyle='-', color='b', label='Single-solve Residual')
                for maxiter in max_IR_iter_list:
                    iter_index = max_IR_iter_list.index(maxiter)
                    plt.plot(range(len(IR_data[iter_index])), IR_data[iter_index], linestyle='-', label=f'IR-{maxiter}iter Residual')
                # plt.plot([0,5], [1,1e-3], linestyle='--', color='k', label='Single-solve Residual')
                plt.xlabel('Shots')
                plt.ylabel('Residual')
                plt.yscale('log')
                plt.xticks(range(len(shots_list)),shots_list)
                plt.title('Effect of shots/IR-iter on residuals')
                plt.grid(True)
                plt.legend()
                # plt.savefig(f'Experiments/IRvsShots-size{self.problem_size}.png')
                plt.show()
                
            else:
                data = shots_data

                plt.figure(figsize=(6,5))

                im = plt.imshow(data, aspect='auto', cmap='viridis')

                plt.xlabel("IR Iteration")
                plt.ylabel("Shots")
                plt.title(f"{self.problem_size}x{self.problem_size} System")

                plt.xticks(np.arange(data.shape[1]))
                # plt.yticks(np.arange(data.shape[0]), labels=np.arange(1, data.shape[0]+1))
                plt.yticks(np.arange(data.shape[0]), labels=shots_list)

                # Reverse y-axis so that 1 is at the bottom
                plt.gca().invert_yaxis()

                cbar = plt.colorbar(im)
                cbar.set_label("Residual")

                # plt.savefig(f"{self.problem_size}x{self.problem_size}_QPE")

                plt.tight_layout()
                plt.show()