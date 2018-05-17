# -*- coding: utf-8 -*-
"""
This module contains the the main controller algorithm.
"""

import numpy as np 
import cvxpy as cvx 

class Controller(object):
	"""
	main controller object
	"""

	def __init__(self, resource_list=[], mu=1):
		self.resource_list = resource_list
		resource_names = []
		for resource in resource_list:
			resource_names.append(resource.name)
		self.resource_names = resource_names
		self.mu = mu
		self.N = len(self.resource_list)

	def addResource(self,resource):
		self.resource_list.append(resource)
		self.resource_names.append(resource.name)
		self.N = len(self.resource_list)

	def solveStep(self,agg_point):
		"""
		takes the aggregated set point and 
		outputs power operating point for each resource and objective value
		"""
		# number of resources
		N = self.N

		# define cvx variables
		p = cvx.Variable(N)
		eps = cvx.Variable(1)

		# define aggragate tracking objective and constraint
		obj = [self.mu*eps]
		constraints = [cvx.sum_entries(p) <= agg_point + eps,
						agg_point - eps <= cvx.sum_entries(p),
						eps >= 0]

		# gather all resources objective function and constraints
		for i in range(N):
			obj_part = self.resource_list[i].costFunc(p[i])
			obj.append(obj_part)
			
			constraints_part = self.resource_list[i].convexHull(p[i])
			constraints.extend(constraints_part)

		# form and solve problem
		obj_final = cvx.Minimize( sum(obj) )
		prob = cvx.Problem(obj_final, constraints)
		prob.solve()

		if prob.status != 'optimal':
			print('Problem status is: ',prob.status)
			p_out = p.value
		else:
			p_out = p.value

		return p_out, eps.value, prob.value

if __name__ == '__main__':
	from resources import *

	# define constants
	mu = 100
	agg_point = np.array([20, 20, 0, -10])

	# define resources
	tcl = TCL('tcl1')
	pv = PVSys('pv1', data=np.array([10, 20, 20, 0, 0]))
	batt1 = Battery('batt1')

	# make controller
	contr = Controller(mu=mu)
	
	# add resources
	contr.addResource(tcl)
	contr.addResource(pv)
	contr.addResource(batt1)

	print('resource names: ',contr.resource_names)

	for t in range(len(agg_point)):
		# solve optimization for a single step
		p_star, eps_star, opt_val = contr.solveStep(agg_point[t])
		print(p_star)
		print(eps_star)
		print(opt_val)

		if contr.N == 1:
			p_operating = contr.resource_list[0].projFeas(p_star)
		else:
			p_operating = np.zeros((contr.N,1))
			for i in range(contr.N):
				p_operating[i,:] = contr.resource_list[i].projFeas(p_star[i,:])

		print('actual operating power:', p_operating)
		print('battery SoC: ', contr.resource_list[2].SoC)



	"""
	Notes:
		Resource comparing if data is string first to prevent elementwise comparison warning
		
		Battery discharging should be positive since positive is for generation and negative is for consumption
			Just switch pmin and pmax since SoC is already made so discharging is positive
		
		make p_last = np.nan instead of None to remove future warning about comparison to None
		self.locked needs to be set to self.locked_next after doing the projection or the hull on the next step will not have been updated


		Artifact of time delay:
		PV wants to output 20 but thinks it can only output 10 since current time max is 10
		Next time step: turns out it could have done 20. still only outputs 10.
		Is there any way to improve this or is this just an inevitable consequence of stochastic nature?
			We don't want the convex opt to think there is extra flexibility when there will not be
			We don't want the actual output to be limited if it turns out there is capability
		First thought: Probably no simple way around this since convex opt will change other devices we do not want to change PV at the last minute
	"""


