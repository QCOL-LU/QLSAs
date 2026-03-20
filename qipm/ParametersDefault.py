import numpy as np
import sys


#===========================================================================
# Linear optimization problem class 
#===========================================================================
class Parameters():

	#-----------------------------------------------------------------------
	# Initialize the problem
	#-----------------------------------------------------------------------
	def __init__(self):

		#-------------------------------------------------------------------
		# Default values of problem generator
		#-------------------------------------------------------------------
		self.Problem_Type 		= "LO"
		self.norm_x 			= 1
		self.norm_s 			= 1
		self.norm_y 			= 1 
		self.norm_b 			= -1
		self.norm_c 			= -1
		self.norm_A				= 1 

		self.decimals			= 3 
		self.condition_number	= 1

		self.seed 				= 669178

		self.has_interior		= True
		self.has_optimal		= True

		self.make_psd			= False 
		self.do_print			= False 
		self.qlsa_print			= False 
		self.symmetry			= False

		self.time_limit 		= 100