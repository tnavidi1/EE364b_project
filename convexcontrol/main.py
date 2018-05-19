# -*- coding: utf-8 -*-
"""
This module contains the the main controller algorithm.
"""

import numpy as np 
import cvxpy as cvx
import matplotlib.pyplot as plt
import time

class Controller(object):
	"""
	main controller object
	"""

	def __init__(self, resource_list=[], mu=100):
		self.resource_list = resource_list
		resource_names = []
		for resource in resource_list:
			resource_names.append(resource.name)
		self.resource_names = resource_names
		self.mu = mu
		self.N = len(self.resource_list)
		self.err = np.zeros((self.N,1))

	def addResource(self,resource):
		self.resource_list.append(resource)
		self.resource_names.append(resource.name)
		self.N = len(self.resource_list)
		self.err = np.zeros((self.N,1))

	def solveStep(self,agg_point,solver='ECOS'):
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
		prob.solve(solver=solver)

		if prob.status != 'optimal':
			print('Problem status is: ',prob.status)
			p_out = p.value
		else:
			p_out = p.value

		return p_out, eps.value, prob.value

	def updateError(self, p_conv, p_operating):
		self.err = self.err + p_operating - p_conv

	def getProjectionsWithError(self,p_conv):
		if self.N == 1:
			p_operating = self.resource_list[0].projFeas(p_conv-self.err)
		else:
			p_operating = np.zeros((self.N,1))
			for i in range(self.N):
				p_operating[i,:] = self.resource_list[i].projFeas(p_conv[i,:] - self.err[i,:])

		return p_operating

	def getProjectionsNoError(self,p_conv):
		if self.N == 1:
			p_operating = self.resource_list[0].projFeas(p_conv)
		else:
			p_operating = np.zeros((self.N,1))
			for i in range(contr.N):
				p_operating[i,:] = self.resource_list[i].projFeas(p_conv[i,:])

		return p_operating

if __name__ == '__main__':
	from resources import *

	# define constants
	mu = 10000
	agg_point = np.concatenate((np.zeros(10),10*np.ones(30),-6*np.arange(10)+10,-50*np.ones(15),15*np.ones(7),np.zeros(8),10*np.ones(20),-10*np.ones(99)))
	T = len(agg_point)

	# define resources
	tcl = TCL('tcl1')
	pv = PVSys('pv1')
	batt1 = Battery('batt1', initial_SoC=0.5,target_SoC=0.2)

	# make controller
	contr = Controller(mu=mu)
	
	# add resources
	contr.addResource(tcl)
	contr.addResource(pv)
	contr.addResource(batt1)

	print('resource names: ',contr.resource_names)
	print('total time horizon: ', T)

	p_conv_all = np.zeros((contr.N,T))
	p_op_all = np.zeros((contr.N,T))
	eps_conv_all = np.zeros((1,T))

	start_time = time.time()

	for t in range(len(agg_point)):
		# solve optimization for a single step
		p_conv, eps_conv, opt_val = contr.solveStep(agg_point[t])

		# get projections onto feasible set
		p_operating = contr.getProjectionsWithError(p_conv)

		# update error term
		contr.updateError(p_conv,p_operating)

		# place data into arrays
		p_conv_all[:,t] = p_conv.flatten()
		p_op_all[:,t] = p_operating.flatten()
		eps_conv_all[:,t] = eps_conv

	print('total comp time: ',time.time() - start_time)

	plt.figure()
	plt.plot(np.sum(p_op_all,axis=0))
	plt.plot(agg_point)
	plt.figure()
	plt.plot(np.sum(p_conv_all,axis=0))
	plt.plot(agg_point)
	plt.show()


