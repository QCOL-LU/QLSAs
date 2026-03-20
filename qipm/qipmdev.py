# from cudaq_qlsa.qlsa.hhl import HHL
# from cudaq_qlsa.noise_model import NoiseModeler
# from cudaq_qlsa.executer import Executer
# from cudaq_qlsa.post_processor import Post_Processor
# from cudaq_qlsa.solver import QuantumLinearSolver
# from cudaq_qlsa.refiner import Refiner

from qlsas.algorithms.hhl.hhl import HHL
from qlsas.data_loader import StatePrep
from qlsas.transpiler import Transpiler
from qlsas.executer import Executer
from qlsas.post_processor import Post_Processor
from qlsas.solver import QuantumLinearSolver
from qlsas.refiner import Refiner
from qlsas.ibm_options import IBMExecutionOptions
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator

import numpy as np
import math
import time
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

service = QiskitRuntimeService(name="QLSAs")

def qipmadaptive(A,Bidx,b,c,
                 x0,y0,s0,
                 n,m,
                 neighborhood,
                 beta,gamma,ir_precision,precision,
                 use_simulator=True):
    
    '''
    interior point method for solving Linear Optimization Problem (A,b,c) with initial interior solution (x0,y0,s0) 
    n variables and m constraints 
    syst= 1 Normal Equations System (NES), 2 Modified Normal Equations System (MNES)
    prec=10**(-6) 
    '''
    
    # noisemodeler = NoiseModeler()
    # noise_config = [0.2, 0.2, 'depolarization', 'depolarization']
    # noise_model  = noisemodeler.noise_modeler(noise_config)
    executer = Executer()
    post_processor = Post_Processor()
    
    x = x0
    s = s0
    y = y0

    mu = np.dot(x.T,s)/n

    print(f'Initial Proximity:{format(np.linalg.norm(x*s - mu*np.ones(n)), ".2E")}, with bound of {format(gamma*mu, ".2E")}')

    cond     = []
    res      = []
    pfeas    = []
    dfeas    = []
    QCalls   = 0
    
    TimeInfo          = {}
    TimeInfo['SB']    = 0
    TimeInfo['CBS']   = 0
    TimeInfo['LS']    = 0

    pfeas.append(np.linalg.norm(A@x-b))
    dfeas.append(np.linalg.norm(A.T@y+s-c))
    
    it=0

    A_B = A[:,Bidx]
    Ahat = np.linalg.solve(A_B, A)                    
    bhat = np.linalg.solve(A_B, b)  

    while (np.dot(x.T,s)>precision):
        print(f'### Iteration {it+1} ###')
        comp=x.T @ s
        mu=(comp.item())/n
        print("mu:", format(mu,".2E"))
        res.append(mu)

        SBstart = time.time()  

        d    = np.sqrt(x) / np.sqrt(s)
        d2   = d * d
        sinv = 1.0 / s

        dB     = d[Bidx]
        dB_inv = 1.0 / dB
        
        # Mhat = (D_Binv@Ahat)@((D@D)@(Ahat.T@D_Binv))
        # L = D_Binv @ Ahat
        L = dB_inv[:, None] * Ahat

        # R = Ahat.T @ D_Binv
        R = Ahat.T * dB_inv[None, :]

        # (D@D) @ R 
        DR = d2[:, None] * R

        # Mhat = L @ DR  
        Mhat = L @ DR

        # sigmahat = (D_Binv@bhat) - beta*mu*D_Binv@((Ahat@Sinv)@ones(n))
        # (D_Binv @ bhat)
        term1 = dB_inv * bhat

        # (Ahat @ Sinv) @ ones(n)
        term2 = Ahat @ sinv

        # D_Binv @ term2
        term2 = dB_inv * term2
        
        # Adaptive Reduction
        if it == 0 or alphasave is None:
            beta = beta
        else:
            if 0.5<=alphasave<=1:
                beta = 0.1
            elif 0.03125<=alphasave<0.5:
                beta = 0.5
            else:
                beta = 0.9

        sigmahat = term1 - beta * mu * term2

        SBend  = time.time()
        SBtime = SBend - SBstart

        print(f'System Built in {SBtime:.4f} seconds with beta:{beta}.')
        TimeInfo['SB'] += SBtime
        
        ##############################
        ####### Quantum Solve! #######
        ##############################
        qMhat     = Mhat / np.linalg.norm(sigmahat)
        qsigmahat = sigmahat / np.linalg.norm(sigmahat)

        print(f'Cond(Mhat): {format(np.linalg.cond(Mhat), ".2E")}')
        cond.append(np.linalg.cond(Mhat))
        
        if not np.allclose(qMhat, qMhat.T.conjugate()):
            raise ValueError("Mhat must be Hermitian")

        ######## Building Circuit #########    
        CBSstart = time.time()
        # initialize qlsa
        hhl = HHL(
            state_prep=StatePrep(method='default'),
            readout = 'measure_x',
            num_qpe_qubits = int(math.log2(len(qsigmahat))), # sigmahat is currently 2**k
            eig_oracle = 'classical'
            )

        # Select backend: simulator for testing, IBM hardware for production
        if use_simulator:
            backend = AerSimulator()
            print(f'Using AerSimulator backend')
        else:
            backend = service.backend("ibm_fez")
            print(f'Using IBM backend: ibm_fez')
        
        hhl_solver = QuantumLinearSolver(
                                        qlsa = hhl,
                                        backend = backend,
                                        target_successful_shots = 1024,
                                        shots_per_batch = 5000,
                                        optimization_level = 3
                                        )
        
        refiner = Refiner(A = qMhat, b = qsigmahat, solver = hhl_solver)
        qlsa_sol = refiner.refine(precision = ir_precision, max_iter = 10, verbose= False, plot=False, open_session=False)
        
        CBSend  = time.time()
        CBStime = CBSend - CBSstart
        print(f'QC Built and System Solved in {CBStime:.4f} seconds.')
        TimeInfo['CBS'] += CBStime
        
        print(f'Refiner terminated within {qlsa_sol['total_iterations']} iterations, at error {format(qlsa_sol['residuals'][-1], ".2E")}.')
        QCalls = QCalls + qlsa_sol['total_iterations']
        
        zz = qlsa_sol['refined_x']

        rhat = Mhat @ zz - sigmahat  

        # dy = (D_Binv@inv(A_B)).T @ zz
        tmp = dB_inv * zz
        dy  = np.linalg.solve(A_B.T, tmp)
        
        # v_B = D_B @ rhat 
        v_B = dB * rhat
        
        # v = [v_B, v_N]
        v = np.zeros(n) 
        v[Bidx] = v_B
        
        # ds = -(A.T @ dy) 
        ds = -(A.T @ dy)
        
        # dx = beta*mu*Sinv@ones(n) - x - (D@D)@ds - v
        dx = beta * mu * sinv - x - (d2 * ds) - v
        
        # Linesearch
        alpha = 1

        x_rec = x + alpha*dx
        s_rec = s + alpha*ds
        y_rec = y + alpha*dy 

        mu_temp = np.dot(x_rec.T,s_rec)/n

        # Line search
        LSstart = time.time()
        
        if neighborhood == 'Large':
            while not np.all(x_rec*s_rec >= gamma * mu_temp * np.ones(n)):
                alpha = 0.5*alpha
                if alpha < 1e-8:
                    raise ValueError('alpha diminishing too fast!')
                    
                x_rec = x + alpha * dx
                s_rec = s + alpha * ds
                y_rec = y + alpha * dy
                
                mu_temp = np.dot(x_rec.T,s_rec)/n
                
        elif neighborhood == 'Small':
            while np.linalg.norm((x_rec*s_rec)/mu_temp - np.ones(n)) > gamma:
                alpha = 0.5*alpha
                if alpha < 1e-8:
                    raise ValueError('alpha diminishing too fast!')
                    
                x_rec = x + alpha * dx
                s_rec = s + alpha * ds
                y_rec = y + alpha * dy
                
                mu_temp = np.dot(x_rec.T,s_rec)/n
        else:
            pass
            # raise ValueError('Invalid Neighborhood Type.')
            
        LSend = time.time()
        LStime = LSend - LSstart
        print(f'LS terminated with alpha = {format(alpha, ".2E")} in {LStime:.4f} seconds.')
        TimeInfo['LS'] += LStime

        alphasave = alpha
        
        x=x+alpha*dx
        s=s+alpha*ds
        y=y+alpha*dy

        pfeas.append(np.linalg.norm(A@x-b))
        dfeas.append(np.linalg.norm(A.T@y+s-c))
        
        comp=x.T@s
        mu=(comp.item())/n

        it=it+1
    
    return (x,s,y,cond,res,pfeas,dfeas,QCalls,TimeInfo)