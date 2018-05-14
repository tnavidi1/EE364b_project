# -*- coding: utf-8 -*-
"""
This module contains energy resource classes.
"""

import numpy as np


class Resource(object):

    """Resource Base Class

    Generic base class for all energy resources modeled in this project

    """

    def __init__(self, name, consumer, producer, cost_function, convex_hull, projection):
        """

        :param name: (str) the name of the resource
        :param consumer: (bool) if true, the resource can consume power
        :param producer: (bool) if true, the resource can produce power
        :param cost_function: (func) a function that accepts a cvxpy.Variable instance and returns a cvxpy Expression
        :param convex_hull: (func) a function that accepts a cvxpy.Variable instance and returns a list of expressions
                            representing the convex hull of that resources feasible set
        :param projection: (func) a function that accepts a setpoint (float) and returns the projection of that
                            setpoint onto the actual feasible set of that resource
        """
        self.name = name
        self.consumer = consumer
        self.producer = producer
        self.cost_function = cost_function
        self.convex_hull = convex_hull
        self.projection = projection

    def convexHull(self, cvxvar):
        return self.convex_hull(cvxvar)

    def costFunc(self, cvxvar):
        return self.cost_function(cvxvar)

    def projFeas(self, setpoint):
        return self.projection(setpoint)


class PVSys(Resource):

    """ PV System Class

	This class represents a solar PV generator. It can accept an pre-recorded power signal or can generate power
	randomly, as a worst-case scenario of the stochastic nature of the resource.

	"""

    def __init__(self, name, Cpv=10, data='random', pmax=30, T=200):
        """

        :param name: (str) the name of the resource
        :param Cpv:  (float) the constant associated with PV cost function
        :param data: a 1-D array or similar representing a PV power signal (optional)
        :param pmax: if data is set to 'random', this is the maximum possible power output of the system
        :param T: if data is set to 'random' this is the number of random data point to generate
        """
        if data == 'random':
            self.power_signal = np.random.uniform(0, pmax, T)
        else:
            self.power_signal = np.squeeze(np.array(data))
        self.t = 0
        self.Cpv = Cpv
        cost_function = lambda x: -Cpv * x
        convex_hull = lambda x: [x >= 0, x <= self.power_signal[self.t]]
        self.projection = lambda x: np.clip(x, 0, self.power_signal[self.t])
        consumer = False
        producer = True
        Resource.__init__(self, name, consumer, producer, cost_function, convex_hull, projection)

    def convexHull(self, x):
        hull = lambda x: [x >= 0, x <= self.power_signal[self.t]]
        self.t += 1
        return hull

    def projFeas(self, setpoint):
        proj = lambda x: np.clip(x, 0, self.power_signal[self.t])
        return proj