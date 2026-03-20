import numpy as np
from copy import deepcopy

#===========================================================================
# Linear optimization problem with a known interior solution 
#===========================================================================
def GenLOwInt(m, n, parameters):

	np.random.seed(parameters.seed)	

	int_x 		= np.random.rand(n)
	int_s 		= 1/int_x # np.random.rand(n)
	int_y 		= np.random.rand(m) - 0.5

	A 			= np.random.rand(m, n) - .5 

	u, s, v 	= np.linalg.svd(A, full_matrices=False)
	s 			= np.linspace(parameters.norm_A/parameters.condition_number, parameters.norm_A, min(m, n))

	A 			= np.dot(u * s, v)

	b 			= np.matmul(A, int_x)
	c 			= np.matmul(A.T, int_y) + int_s

	return A, b, c, int_x, int_y, int_s

def GenLOwIntOpt(m,n,B,N,parameters):

    #B=m-1
    #N=n-m-1
    norm_b          = -1
    norm_c          = -1
    norm_x          = 1
    norm_y          = 1
    norm_s          = 1
    
    mask            = np.block([np.ones(B),-1*np.ones(N)])
    maskB           = [1 if mask[i]==1 else 0 for i in range(n)]
    maskN           = [1 if mask[i]==-1 else 0 for i in range(n)]
    
    opt_x 			= np.ones(n)
    opt_x 			= np.multiply(opt_x,maskB)
    opt_s 			= np.ones(n)
    opt_s 			= np.multiply(opt_s,maskN)
    opt_y 			= np.ones(m) - 0.5
    
    A		        = np.random.rand(m, n) - 0.5
    u, s, v         = np.linalg.svd(A, full_matrices=False)
    s 	            = np.linspace(parameters.norm_A/parameters.condition_number, parameters.norm_A, min(m, n))
    A 	            = np.dot(u * s, v) 
    A[:m,:m]        = np.eye(m)
    
    int_x 			= np.ones(n+1)
    int_s 			= np.ones(n+1) # np.ones(n+1) #(1/n)/int_x
    int_y 			= np.zeros(m+1)
    int_y[m]	    = 0.5
    
    delta			= np.dot((int_x[:n]-opt_x).T,(int_s[:n]-opt_s))
    de 			    = max(0,-delta/int_x[n])
    int_s[n] 		= de +0.1 if int_s[n] <= de else int_s[n]
    
    opt_x 			= np.append(opt_x,0)
    opt_s 			= np.append(opt_s,(delta/int_x[n])+int_s[n])
    opt_y 			= np.append(opt_y,0)
    
    
    ahat 			= (1/int_x[n])*(np.dot(A,(opt_x[:n]-int_x[:n])))
    dhat 			= (1/int_y[m])*(np.dot(A.T,(opt_y[:m]-int_y[:m]))+opt_s[:n]-int_s[:n])
    dlast 		    = (1/int_x[n])*(np.dot(dhat,(opt_x[:n]-int_x[:n])))
    AA 			    = np.block([[A,               np.matrix(ahat).T],
                                [np.matrix(dhat), np.matrix(dlast)]])
    
    
    b 				= np.dot(AA, opt_x)
    c 				= np.dot(AA.T, opt_y) + opt_s
    
    BB = []
    for i in range(n):
        if maskB[i] == 1:
            BB.append(i)
    results     	= (AA, BB, b, c, opt_x, opt_y, opt_s, int_x, int_y, int_s)
    print("Basis:                                    ", BB)
    print("Condition number of A:                    ", format(np.linalg.cond(AA), ".2E"))
    print("Condition number of A_B:                  ", format(np.linalg.cond(A[:,BB]), ".2E"))
    return results



