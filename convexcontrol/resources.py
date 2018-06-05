# -*- coding: utf-8 -*-
"""
This module contains energy resource classes.
"""

import numpy as np
import cvxpy as cvx


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
        # output should be a single item not inside a list
        if self.cost_function is not None:
            return self.cost_function(cvxvar)

    def convexHull(self, cvxvar):
        # Output should be inside a list
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
        if isinstance(data, str):
            if data == 'random':
                self.power_signal = np.random.uniform(0, pmax, T)
                self.H = T
        else:
            self.power_signal = np.squeeze(np.array(data))
            self.H = len(self.power_signal)
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
        proj = np.clip(setpoint, 0, self.power_signal[min(self.t, self.H-1)])  # last projection index
        return proj

class PVSysR2(Resource):

    """ PV System Class in R2

	This class represents a solar PV generator with real and reactive power control. It can accept an pre-recorded power
	signal or can generate power randomly, as a worst-case scenario of the stochastic nature of the resource.

	"""

    def __init__(self, name, Cpv=10, data='random', pmax=30, T=200):
        """

        :param name: (str) the name of the resource
        :param Cpv:  (float) the constant associated with PV cost function
        :param data: a 1-D array or similar representing a PV power signal (optional)
        :param pmax: the maximum possible power output of the system, defines the feasible set in R2
        :param T: if data is set to 'random' this is the number of random data point to generate
        """
        if isinstance(data, str):
            if data == 'random':
                self.power_signal = np.random.uniform(0, pmax, T)
                self.H = T
        else:
            self.power_signal = np.squeeze(np.array(data))
            self.H = len(self.power_signal)
        self.pmax = pmax
        self.t = 0
        self.Cpv = Cpv
        cost_function = lambda x: -Cpv * x[0]
        consumer = False
        producer = True
        Resource.__init__(self, name, consumer, producer, cost_function)

    def convexHull(self, cvxvar):
        hull = [
            cvxvar[0] >= 0,
            cvxvar[0] <= self.power_signal[self.t],
            cvx.norm(cvxvar, 2) <= self.pmax
        ]
        self.t += 1
        return hull

    def projFeas(self, setpoint):
        proj0 = np.clip(setpoint[0], 0, self.power_signal[min(self.t, self.H-1)])
        if np.linalg.norm((proj0, setpoint[1])) <= self.pmax:
            proj = np.array((proj0, setpoint[1]))
        else:
            proj1 = np.sign(setpoint[1]) * np.sqrt(self.pmax ** 2 - proj0 ** 2)
            proj = np.array((proj0, proj1))
        return proj

class Battery(Resource):

    """ Battery Class

    This implements a simple battery model. The state of charge is estimated by power input/output.

    """


    def __init__(self, name, Cb=10, Cbl=0, pmin=-50, pmax=50, initial_SoC=0.2, target_SoC=0.5, capacity=30, eff=0.95,
                 tstep=1./60, T=200):
        if isinstance(target_SoC, float):
            self.target_SoC = np.ones(T) * target_SoC
            self.H = T
        else:
            self.target_SoC = target_SoC
            self.H = len(target_SoC)
        consumer = True
        producer = True
        self.Cb = Cb
        self.Cbl = Cbl
        self.pmax = pmax
        self.pmin = pmin
        self.SoC = initial_SoC
        self.SoC_next = initial_SoC
        self.capacity = capacity
        self.eff = eff
        self.tstep = np.float(tstep)
        self.t = 0
        Resource.__init__(self, name, consumer, producer)

    def costFunc(self, cvxvar):
        p_soc = np.abs(self.SoC - self.target_SoC[self.t]) * self.capacity * self.eff / self.tstep
        if self.SoC >= self.target_SoC[self.t]:
            p = min(self.pmax, p_soc)
            cost = self.Cb * np.power(cvxvar - p, 2)    # if above the desired SoC, try to discharge
        else:
            p = max(self.pmin, -p_soc)
            cost = self.Cb * np.power(cvxvar - p, 2)    # if below the desired SoC, try to charge
        return cost + self.Cbl * cvx.abs(cvxvar)         # Cbl represents cost of the battery amortized over
                                                        # total lifetime energy

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
        self.t += 1
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

class BatteryR2(Resource):

    """ Battery Class

    This implements a simple battery model. The state of charge is estimated by power input/output.

    """


    def __init__(self, name, Cb=10, Cbl=0, pmin=-50, pmax=50, initial_SoC=0.2, target_SoC=0.5, capacity=30, eff=0.95,
                 tstep=1./60):
        if isinstance(target_SoC, float):
            self.target_SoC = np.ones(T) * target_SoC
            self.H = T
        else:
            self.target_SoC = target_SoC
            self.H = len(target_SoC)
        consumer = True
        producer = True
        self.Cb = Cb
        self.Cbl = Cbl
        self.pmax = pmax
        self.pmin = pmin
        self.SoC = initial_SoC
        self.SoC_next = initial_SoC
        self.capacity = capacity
        self.eff = eff
        self.tstep = np.float(tstep)
        self.t = 0
        Resource.__init__(self, name, consumer, producer)

    def costFunc(self, cvxvar):
        p_soc = np.abs(self.SoC - self.target_SoC[self.t]) * self.capacity * self.eff / self.tstep
        if self.SoC >= self.target_SoC[self.t]:
            p = min(self.pmax, p_soc)
            cost = self.Cb * np.power(cvxvar[0] - p, 2)     # if above the desired SoC, try to discharge
        else:
            p = max(self.pmin, -p_soc)
            cost = self.Cb * np.power(cvxvar[0] - p, 2)     # if below the desired SoC, try to charge
        return cost + self.Cbl * cvx.abs(cvxvar[0])          # Cbl represents cost of the battery amortized over
                                                            # total lifetime energy

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
        self.t += 1
        return [
            cvxvar[0] >= pmin,
            cvxvar[0] <= pmax,
            cvx.norm(cvxvar, 2) <= self.pmax
        ]

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
        sp0 = np.clip(setpoint[0], pmin, pmax)
        self.SoC_next = self.SoC - sp0 * self.tstep / (self.capacity * self.eff)
        if np.linalg.norm((sp0, setpoint[1])) <= self.pmax:
            sp = np.array((sp0, setpoint[1]))
        else:
            sp1 = np.sign(setpoint[1]) * np.sqrt(self.pmax ** 2 - sp0 ** 2)
            sp = np.array((sp0, sp1))
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
        if isinstance(p_con, str):
            if p_con == 'simple':
                self.p_con = np.zeros(T)
                self.p_con[int(T/4):int(T/2)] = 1
                self.p_con[int(T/2):int(3*T/4)] = 2
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
        self.p_last = np.nan
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
            # Note that the 'max' power is a consumption and therefore self.step_size is a negative value
            hull = [cvxvar <= np.min(self.states) * self.step_size, cvxvar >= np.max(self.states) * self.step_size]
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
        #self.locked = self.locked_next
        return sp


class PVSysR2(Resource):

    """ PV System Class in R2
    This class represents a solar PV generator with real and reactive power control. It can accept an pre-recorded power
    signal or can generate power randomly, as a worst-case scenario of the stochastic nature of the resource.
    """

    def __init__(self, name, Cpv=10, data='random', pmax=30, T=200):
        """
        :param name: (str) the name of the resource
        :param Cpv:  (float) the constant associated with PV cost function
        :param data: a 1-D array or similar representing a PV power signal (optional)
        :param pmax: the maximum possible power output of the system, defines the feasible set in R2
        :param T: if data is set to 'random' this is the number of random data point to generate
        """
        if isinstance(data, str):
            if data == 'random':
                self.power_signal = np.random.uniform(0, pmax, T)
        else:
            self.power_signal = np.squeeze(np.array(data))
        self.pmax = pmax
        self.t = 0
        self.Cpv = Cpv
        cost_function = lambda x: -Cpv * x[0]
        consumer = False
        producer = True
        Resource.__init__(self, name, consumer, producer, cost_function)

    def convexHull(self, cvxvar):
        hull = [
            cvxvar[0] >= 0,
            cvxvar[0] <= self.power_signal[self.t],
            cvx.norm(cvxvar, 2) <= self.pmax
        ]
        self.t += 1
        return hull

    def projFeas(self, setpoint):
        proj0 = np.clip(setpoint[0], 0, self.power_signal[self.t])
        if np.linalg.norm((proj0, setpoint[1])) <= self.pmax:
            proj = np.array((proj0, setpoint[1]))
        else:
            proj1 = np.sign(setpoint[1]) * np.sqrt(self.pmax ** 2 - proj0 ** 2)
            proj = np.array((proj0, proj1))
        return proj


class BatteryR2(Resource):

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
                cost = self.Cb * np.power(x[0] - self.pmax, 2) # if above the desired SoC, try to discharge
            else:
                cost = self.Cb * np.power(x[0] - self.pmin, 2) # if below the desired SoC, try to charge
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
        return [
            cvxvar[0] >= pmin,
            cvxvar[0] <= pmax,
            cvx.norm(cvxvar, 2) <= self.pmax
        ]

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
        sp0 = np.clip(setpoint[0], pmin, pmax)
        self.SoC_next = self.SoC - sp0 * self.tstep / (self.capacity * self.eff)
        if np.linalg.norm((sp0, setpoint[1])) <= self.pmax:
            sp = np.array((sp0, setpoint[1]))
        else:
            sp1 = np.sign(setpoint[1]) * np.sqrt(self.pmax ** 2 - sp0 ** 2)
            sp = np.array((sp0, sp1))
        return sp

class DiscreteR2(Resource):
    """ Discrete operating modes device
    Device can output power from a collection of any points
    """

    def __init__(self, name, points=np.array([[-10,-5], [-20, -10], [-30, -20]]), desired=np.zeros(200,dtype=int),
                 Cdisc=10, t_lock=5):
        from scipy.spatial import ConvexHull

        points = np.array(points, dtype=float) # set of points in np array shape (points, dim=2)
        self.points = points
        self.desired = desired # list of desired operating points ex. [1, 2, 4, 0] corresponds to row of points array
        modes, dim = points.shape
        self.modes = modes # number of operating mfodes (points)

        if modes > 2:
            try:
                ch = ConvexHull(points) # make convex hull object of points
                hull_mat = ch.equations
                self.A = hull_mat[:,0:2]
                self.b = hull_mat[:,2]
                self.set_dim = 2 # 2 for polygon, 1 for line, 0 for point
            except:
                print('Discrete resource points lie on a line')
                self.set_dim = 1
                ind1 = np.argmin(points[:,0])
                ind2 = np.argmax(points[:,0])
                p1 = points[ind1,:]
                p2 = points[ind2,:]
                self.min0 = np.minimum(p1[0],p2[0])
                self.max0 = np.maximum(p1[0],p2[0])
                self.c = (p2[1]-p1[1])/(p2[0] - p1[0])
                self.a = p1[0]
                self.b = p1[1]

        elif modes == 2:
            self.set_dim = 1
            p1 = points[0,:]
            p2 = points[1,:]
            self.min0 = np.minimum(p1[0],p2[0])
            self.max0 = np.maximum(p1[0],p2[0])
            self.c = (p2[1]-p1[1])/(p2[0] - p1[0])
            self.a = p1[0]
            self.b = p1[1]
        else:
            self.set_dim = 0 # for point

        consumer = True
        producer = False
        self.t = 0
        self.Cdisc = Cdisc
        self.t_lock = t_lock
        self.timer = 0
        self.locked = False
        self.locked_next = False
        self.p_last = np.array([np.nan, np.nan])
        Resource.__init__(self, name, consumer, producer)

    def costFunc(self, cvxvar):
        """
        Cost is just scaled euclidean distance of variable from desired operating point
        """
        cost = self.Cdisc * cvx.norm(cvxvar - self.points[self.desired[self.t],:], 2)
        return cost

    def convexHull(self, cvxvar):
        if not self.locked:
            if self.set_dim == 2:
                hull = [self.A*cvxvar + self.b <= 0]
            elif self.set_dim == 1:
                hull = [cvxvar[0] >= self.min0, cvxvar[0] <= self.max0, cvxvar[1] == self.c*(cvxvar[0]-self.a) + self.b]
            else:
                hull = [cvxvar == self.points]
        else:
            hull = [cvxvar == self.p_last]
        # update internal state
        self.locked = self.locked_next
        self.t += 1
        return hull

    def projFeas(self, setpoint):
        if not self.locked:
            # K nearest neighbors
            dist_old = np.infty
            for i in range(self.modes):
                dist = np.linalg.norm(setpoint - self.points[i,:])
                if dist < dist_old:
                    ind = i
                    dist_old = dist

            sp = self.points[ind,:]

            if sp[0] != self.p_last[0] or sp[1] != self.p_last[1]:
                self.locked_next = True
        else:
            sp = self.p_last
            self.timer += 1
            if self.timer == self.t_lock:
                self.locked_next = False
                self.timer = 0
        self.p_last = sp
        return sp



