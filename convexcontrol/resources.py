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
        projection = lambda x: np.clip(x, 0, self.power_signal[self.t])
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


class Battery(Resource):

    """ Battery Class

    This implements a simple battery model. The state of charge is estimated by power input/output.

    """


    def __init__(self, name, Cb=10, pmin=-50, pmax=50, initial_SoC=0.2, target_SoC=0.5, capacity=30, eff=0.95,
                 tstep=1./60):
        consumer = True
        producer = True
        self.Cb = Cb
        self.pmax = pmax
        self.pmin = pmin
        self.target_SoC = target_SoC
        self.SoC = initial_SoC
        self.Soc_next = None
        self.capacity = capacity
        self.eff = eff
        self.tstep = np.float(tstep)
        def cost_function(x):
            if self.SoC >= self.target_SoC:
                cost = self.Cb * np.power(x - self.pmin, 2)
            else:
                cost = self.Cb * np.power(x - self.pmax, 2)
            return cost
        convex_hull = lambda x: [x >= pmin, x <= pmax]
        projection = lambda x: np.clip(x, self.pmin, self.pmax)
        Resource.__init__(name, consumer, producer, cost_function, convex_hull, projection)

    def convexHull(self, cvxvar):
        """
        The feasible set of power output (input) of the battery is defined not only by the physical limits self.pmin and
        self.pmax, but also by the state of charge of the battery. The battery cannot source more power over a time step
        than it has charge remaining, cannot accept more power over a time step than it has free SoC left. In other
        words:

            0 <= (SoC * capacity) - (power * time_step) / efficiency <= capacity

        Note that we define positive power as generation. The more restrictive constraint is the one that must hold.
        This set is already convex.

        In keeping with the algorithm design, this is the "observation" of the master algorithm, ostensibly obtained
        from the previous setpoint request. So, although the actual SoC changed after the previous setpoint request,
        we assume that the master algorithm does not know this yet.

        :param cvxvar: a cvxpy.Variable instance
        :return: a list of cvxpy constraints
        """
        pmin = max(self.pmin,(self.SoC - 1.) * self.capacity * self.eff / self.tstep)
        pmax = min(self.pmax, self.SoC * self.capacity * self.eff/ self.tstep)
        self.SoC = self.Soc_next
        return [cvxvar >= pmin, cvxvar <= pmax]

    def costFunc(self, cvxvar):
        if self.SoC >= self.target_SoC:
            cost = self.Cb * np.power(cvxvar - self.pmin, 2)
        else:
            cost = self.Cb * np.power(cvxvar - self.pmax, 2)
        return cost

    def projFeas(self, setpoint):
        """
        The feasible set is identical to the description given in self.ConvexHull. After the setpoint is implemented,
        the state of charge of the battery changes according to

        SoC_next = SoC - (power * time_step) / (efficiency * capacity)

        We store this in the attribute self.SoC_next until the master algorithm

        :param setpoint: (float) the requested power output (input) of the battery
        :return:
        """
        pmin = max(self.pmin, (self.SoC - 1.) * self.capacity * self.eff / self.tstep)
        pmax = min(self.pmax, self.SoC * self.capacity * self.eff / self.tstep)
        sp = np.clip(setpoint, pmin, pmax)
        self.Soc_next = self.SoC - sp * self.tstep / (self.capacity * self.eff)
        return sp