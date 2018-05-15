# -*- coding: utf-8 -*-
"""
This module contains energy resource classes.
"""

import numpy as np


class Resource(object):

    """Resource Base Class

    Generic base class for all energy resources modeled in this project

    """

    def __init__(self, name, consumer, producer, cost_function=None, convex_hull=None, projection=None):
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

    def costFunc(self, cvxvar):
        if self.cost_function is not None:
            return self.cost_function(cvxvar)

    def convexHull(self, cvxvar):
        if self.convex_hull is not None:
            return self.convex_hull(cvxvar)

    def projFeas(self, setpoint):
        if self.projection is not None:
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
        consumer = False
        producer = True
        Resource.__init__(self, name, consumer, producer, cost_function)

    def convexHull(self, cvxvar):
        hull = [cvxvar >= 0, cvxvar <= self.power_signal[self.t]]
        self.t += 1
        return hull

    def projFeas(self, setpoint):
        proj = np.clip(setpoint, 0, self.power_signal[self.t])
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
        self.SoC_next = initial_SoC
        self.capacity = capacity
        self.eff = eff
        self.tstep = np.float(tstep)
        def cost_function(x):
            if self.SoC >= self.target_SoC:
                cost = self.Cb * np.power(x - self.pmin, 2) # if above the desired SoC, try to discharge
            else:
                cost = self.Cb * np.power(x - self.pmax, 2) # if below the desired SoC, try to charge
            return cost
        Resource.__init__(self, name, consumer, producer, cost_function)

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
        we assume that the master algorithm does not know this yet. Thus, we update the self.SoC attribute after
        calculating the feasible set based on the old value.

        :param cvxvar: a cvxpy.Variable instance
        :return: a list of cvxpy constraints
        """
        pmin = max(self.pmin,(self.SoC - 1.) * self.capacity * self.eff / self.tstep)
        pmax = min(self.pmax, self.SoC * self.capacity * self.eff/ self.tstep)
        self.SoC = self.SoC_next
        return [cvxvar >= pmin, cvxvar <= pmax]

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
        self.SoC_next = self.SoC - sp * self.tstep / (self.capacity * self.eff)
        return sp


class TCL(Resource):

    """ Thermostatically Controlled Load Class

    This extremely simply TCL model assumes a that the device has a maximum power at time t of p_max[t] and that the
    possible setpoints are discrete and evenly distributed between 0 and p_max[t]. For example, a 2-stage HVAC system
    might have the possible operating states {0kW, 10kW, 20kW}. The set point, p_con, represents the output of
    the local thermostat control. The default class initialization synthesizes a time-series signal for p_con,
    representing a simple thermostat control sequence:


            2 |             ------
    p_con   1 |       ------
            0 | ------            ------
              |_______________________________
                            t

    The cost function for P1 is simply the square error loss between the optimal set point at time t and p_set[t],
    multipylied by the constant Chvac. A large value for Chvac represents a low tolerance by the user to have their
    HVAC settings changed. The feasible set is either {0,1,2} with the convex hull [0, 2], or simply {p_last} if the
    system is locked, where p_last is last implemented power set point. We assume that the system becomes locked after
    each change in implemented setpoint, for a predetermined number of steps, t_lock.

    """

    def __init__(self, name, Chvac=10, steps=2, pmax=-20,  p_con='simple', T=200, t_lock=5):
        if p_con == 'simple':
            self.p_con = np.zeros(T)
            self.p_con[T/4:T/2] = 1
            self.p_con[T/2:3*T/4] = 2
        else:
            self.p_con = np.squeeze(np.array(p_con))
        consumer = True
        producer = False
        self.t = 0
        self.states = range(steps+1)
        self.step_size = pmax * 1. / steps
        self.Chvac = Chvac
        self.t_lock = t_lock
        self.timer = 0
        self.locked = False
        self.locked_next = False
        self.p_last = None
        Resource.__init__(self, name, consumer, producer)

    def costFunc(self, cvxvar):
        """
        Note that the impact of the local thermostat constrol signal is modeled as part of the cost function, not the
        feasible set. We assume that the master algorithm has final say over what is implemented by the system,
        but that the thermostat controller can impact that voice through the cost function.

        :param cvxvar: a cvxpy.Variable instance
        :return: a convex cvxpy expression
        """
        cost = self.Chvac * np.power(cvxvar - self.p_con[self.t] * self.step_size, 2)
        return cost

    def convexHull(self, cvxvar):
        if not self.locked:
            hull = [cvxvar >= np.min(self.states) * self.step_size, cvxvar <= np.max(self.states) * self.step_size]
        else:
            hull = [cvxvar == self.p_last]
        # update internal state
        self.locked = self.locked_next
        self.t += 1
        return hull

    def projFeas(self, setpoint):
        if not self.locked:
            sp = np.round(np.clip(setpoint * 1. /  self.step_size, np.min(self.states), np.max(self.states)), 0)
            sp *= self.step_size
            if sp != self.p_last:
                self.locked_next = True
        else:
            sp = self.p_last
            self.timer += 1
            if self.timer == self.t_lock:
                self.locked_next = False
                self.timer = 0
        self.p_last = sp
        return sp
