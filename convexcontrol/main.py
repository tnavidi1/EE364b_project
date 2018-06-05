# -*- coding: utf-8 -*-
"""
This module contains the the main controller algorithm.
"""

import numpy as np
import pandas as pd
import cvxpy as cvx
import matplotlib.pyplot as plt
import time

from .resources import Battery, TCL

class Controller(object):
    """
    main controller object
    """

    def __init__(self, resource_list=None, mu=100, tstep=5./60):
        resource_names = []
        if resource_list is None:
            self.resource_list = []
        else:
            self.resource_list = resource_list
            for resource in resource_list:
                resource_names.append(resource.name)
        self.resource_names = resource_names
        self.mu = mu
        self.N = len(self.resource_list)
        self.err = np.zeros(self.N)
        self.p_requested = None
        self.eps = None
        self.prob_val = None
        self.output = None
        self.pcc_signal = None
        self.tstep = tstep

    def addResource(self,resource):
        if isinstance(resource, TCL):
            resource.tstep = float(self.tstep)
        self.resource_list.append(resource)
        self.resource_names.append(resource.name)
        self.N = len(self.resource_list)
        self.err = np.zeros(self.N)

    def runSimulation(self, pcc_signal, error_diffusion=True, solver='ECOS'):
        batteries = np.arange(self.N)[[isinstance(r, Battery) for r in  self.resource_list]]
        cols = ['PCC req', 'PCC imp', 'eps']
        cols.extend([n + ' SoC' for n in np.array(self.resource_names)[batteries]])
        cols.extend([r + ' req' for r in self.resource_names])
        cols.extend([r + ' imp' for r in self.resource_names])
        output = pd.DataFrame(columns=cols, index=range(len(pcc_signal)))
        for t in range(len(pcc_signal)):
            self.solveStep(pcc_signal[t], solver=solver)
            output.loc[t]['PCC req'] = np.sum(self.p_requested)
            output.loc[t]['eps'] = self.eps
            if error_diffusion:
                self.getProjectionsWithError()
                self.updateError()
            else:
                self.getProjectionsNoError()
            for i in range(self.N):
                key1 = self.resource_names[i] + ' req'
                key2 = self.resource_names[i] + ' imp'
                output.loc[t][key1] = self.p_requested[i]
                output.loc[t][key2] = self.p_operating[i]

            output.loc[t]['PCC imp'] = np.sum(self.p_operating)
            for j in batteries:
                key = self.resource_names[j] + ' SoC'
                output.loc[t][key] = np.float(self.resource_list[j].SoC)
        self.pcc_signal = pcc_signal
        self.output = output

    def solveStep(self, agg_point, solver='ECOS'):
        """
        takes the aggregated set point and
        outputs power operating point for each resource and objective value
        """
        # number of resources
        N = self.N

        # define cvx variables
        p = cvx.Variable(N)
        eps = cvx.Variable(1)

        # define aggregate tracking objective and constraint
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

        self.p_requested = p_out.A1
        self.eps = eps.value
        self.prob_val = prob.value

    def updateError(self):
        self.err = self.err +self.p_operating - self.p_requested

    def getProjectionsWithError(self):
        p_req = self.p_requested
        p_operating = np.array([self.resource_list[i].projFeas(p_req[i] - self.err[i]) for i in range(self.N)])
        self.p_operating = p_operating

    def getProjectionsNoError(self):
        p_req = self.p_requested
        p_operating = np.array([self.resource_list[i].projFeas(p_req[i]) for i in range(self.N)])
        self.p_operating = p_operating

    def plotReqImpPower(self):
        batteries = np.arange(self.N)[[isinstance(r, Battery) for r in self.resource_list]]
        n_rows = 1 + len(self.resource_names) + len(batteries)
        fig, ax = plt.subplots(nrows=n_rows, sharex=True, figsize=(n_rows*3, 10))
        xs = range(1, len(self.pcc_signal) + 1)
        ax[0].plot(xs, self.output['PCC req'], label='requested')
        ax[0].plot(xs, self.output['PCC imp'], label='implemented')
        ax[0].plot(xs, self.pcc_signal, ls='--', label='set point')
        ax[0].set_title('aggregate set point signal')
        ax[0].set_ylabel('kW')
        ax[0].legend(loc=(1.01, .1))
        counter = 1
        for resource in self.resource_list:
            is_battery = isinstance(resource, Battery)
            is_tcl = isinstance(resource, TCL)
            name = resource.name
            key1 = name + ' req'
            key2 = name + ' imp'
            ax[counter].plot(xs, self.output[key1], label='requested')
            ax[counter].plot(xs, self.output[key2], label='implemented')
            ax[counter].set_title(name + ' power signal')
            ax[counter].set_ylabel('kW')
            if is_tcl:
                ax[counter].plot(xs, resource.p_con * resource.step_size, label='desired')
            ax[counter].legend(loc=(1.01, .1))
            counter += 1
            if is_battery:
                ax[counter].plot(xs, self.output[name + ' SoC'])
                ax[counter].set_title(name + ' SoC')
                ax[counter].set_ylabel('SoC')
                ax[counter].legend(loc=(1.01, .1))
                counter += 1
        return fig

    def plotReqImpTotalEnergy(self):
        n_rows = 1 + len(self.resource_names)
        fig, ax = plt.subplots(nrows=n_rows, sharex=True, figsize=(n_rows * 3, 10))
        xs = range(1, len(self.pcc_signal) + 1)
        ax[0].plot(xs, self.tstep * np.cumsum(self.output['PCC req']), label='requested')
        ax[0].plot(xs, self.tstep * np.cumsum(self.output['PCC imp']), label='implemented')
        ax[0].plot(xs, self.tstep * np.cumsum(self.pcc_signal), ls='--', label='set point')
        ax[0].set_title('total aggregate energy')
        ax[0].set_ylabel('kWh')
        ax[0].legend(loc=(1.01, .1))
        counter = 1
        for resource in self.resource_list:
            name = resource.name
            key1 = name + ' req'
            key2 = name + ' imp'
            ax[counter].plot(xs, self.tstep * np.cumsum(self.output[key1]), label='requested')
            ax[counter].plot(xs, self.tstep * np.cumsum(self.output[key2]), label='implemented')
            ax[counter].set_title(name + ' total energy')
            ax[counter].set_ylabel('kWh')
            ax[counter].legend(loc=(1.01, .1))
            counter += 1
        return fig


class ControllerR2(object):
    """
    main controller object
    """

    def __init__(self, resource_list=None, mu=100, tstep=5./60):
        resource_names = []
        if resource_list is None:
            self.resource_list = []
        else:
            self.resource_list = resource_list
            for resource in resource_list:
                resource_names.append(resource.name)
        self.resource_names = resource_names
        self.mu = mu
        self.N = len(self.resource_list)
        self.err = np.zeros((2,self.N))
        self.p_requested = None
        self.eps = None
        self.prob_val = None
        self.output_real = None
        self.output_reactive = None
        self.pcc_signal = None
        self.tstep = tstep

    def addResource(self,resource):
        if isinstance(resource, BatteryR2):
            resource.tstep = self.tstep
        self.resource_list.append(resource)
        self.resource_names.append(resource.name)
        self.N = len(self.resource_list)
        self.err = np.zeros((2,self.N))

    def runSimulation(self, pcc_signal, error_diffusion=True, solver='ECOS'):
        batteries = np.arange(self.N)[[isinstance(r, Battery) for r in  self.resource_list]]
        cols = ['PCC req', 'PCC imp', 'eps']
        cols.extend([r + ' req' for r in self.resource_names])
        cols.extend([r + ' imp' for r in self.resource_names])
        SoC_cols = [n + ' SoC' for n in np.array(self.resource_names)[batteries]]
        dim, T = pcc_signal.shape
        output_real = pd.DataFrame(columns=cols+SoC_cols, index=range(T))
        output_reactive = pd.DataFrame(columns=cols, index=range(T))
        for t in range(T):
            self.solveStep(pcc_signal[:,t], solver=solver)
            pcc_req = np.sum(self.p_requested,axis=1)
            output_real.loc[t]['PCC req'] = pcc_req[0]
            output_reactive.loc[t]['PCC req'] = pcc_req[1]
            output_real.loc[t]['eps'] = self.eps[0]
            output_reactive[t]['eps'] = self.eps[1]
            if error_diffusion:
                self.getProjectionsWithError()
                self.updateError()
            else:
                self.getProjectionsNoError()
            for i in range(self.N):
                key1 = self.resource_names[i] + ' req'
                key2 = self.resource_names[i] + ' imp'
                output_real.loc[t][key1] = self.p_requested[0,i]
                output_real.loc[t][key2] = self.p_operating[0,i]
                output_reactive.loc[t][key1] = self.p_requested[1, i]
                output_reactive.loc[t][key2] = self.p_operating[1, i]
            pcc_imp = np.sum(self.p_operating, axis=1)
            output_real.loc[t]['PCC imp'] = pcc_imp[0]
            output_reactive.loc[t]['PCC imp'] = pcc_imp[1]
            for j in batteries:
                key = self.resource_names[j] + ' SoC'
                output_real.loc[t][key] = np.float(self.resource_list[j].SoC)
        self.pcc_signal = pcc_signal
        self.output_real = output_real
        self.output_reactive = output_reactive

    def solveStep(self, agg_point, solver='ECOS'):
        """
        takes the aggregated set point real and reactive and
        outputs power operating point for each resource and objective value
        """
        # number of resources
        N = self.N

        # define cvx variables
        p = cvx.Variable(2,N)
        eps = cvx.Variable(2)

        # define aggregate tracking objective and constraint
        obj = [self.mu*cvx.norm(eps)]
        constraints = [cvx.sum_entries(p,axis=1) <= agg_point + eps,
                        agg_point - eps <= cvx.sum_entries(p,axis=1),
                        eps >= 0]

        # gather all resources objective function and constraints
        for i in range(N):
            obj_part = self.resource_list[i].costFunc(p[:,i])
            obj.append(obj_part)

            constraints_part = self.resource_list[i].convexHull(p[:,i])
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

        self.p_requested = np.asarray(p_out)
        self.eps = eps.value
        self.prob_val = prob.value

        #print('eps val', eps.value)
        #print('p requested', self.p_requested)

    def updateError(self):
        self.err = self.err + self.p_operating - self.p_requested

    def getProjectionsWithError(self):
        p_req = self.p_requested
        p_operating = (np.reshape(self.resource_list[i].projFeas(p_req[:,i] - self.err[:,i]), (2,1)) for i in range(self.N))
        self.p_operating = np.hstack(p_operating)

        #print('error', self.err)
        #print('p operating', self.p_operating)

    def getProjectionsNoError(self):
        p_req = self.p_requested
        p_operating = np.array([self.resource_list[i].projFeas(p_req[:,i]) for i in range(self.N)])
        self.p_operating = p_operating

    def plotReqImpPower(self, select='real'):
        batteries = np.arange(self.N)[[isinstance(r, BatteryR2) for r in self.resource_list]]
        if select == 'real':
            output = self.output_real
            ix = 0
            n_rows = 1 + len(self.resource_names) + len(batteries)
        else:
            output = self.output_reactive
            ix = 1
            n_rows = 1 + len(self.resource_names)
        fig, ax = plt.subplots(nrows=n_rows, sharex=True, figsize=(n_rows*3, 10))
        xs = range(1, len(self.pcc_signal) + 1)
        ax[0].plot(xs, output['PCC req'], label='requested')
        ax[0].plot(xs, output['PCC imp'], label='implemented')
        ax[0].plot(xs, self.pcc_signal[ix, :], ls='--', label='set point')
        ax[0].set_title('aggregate set point signal')
        if select == 'real':
            ax[0].set_ylabel('kW')
        else:
            ax[0].set_ylabel('kvar')
        ax[0].legend(loc=(1.01, .1))
        counter = 1
        for resource in self.resource_list:
            is_battery = isinstance(resource, BatteryR2)
            is_tcl = isinstance(resource, DiscreteR2)
            name = resource.name
            key1 = name + ' req'
            key2 = name + ' imp'
            ax[counter].plot(xs, output[key1], label='requested')
            ax[counter].plot(xs, output[key2], label='implemented')
            ax[counter].set_title(name + ' power signal')
            if select == 'real':
                ax[counter].set_ylabel('kW')
            else:
                ax[counter].set_ylabel('kvar')
            if is_tcl:
                ax[counter].plot(xs, resource.p_con * resource.step_size, label='desired')
            ax[counter].legend(loc=(1.01, .1))
            counter += 1
            if is_battery and select == 'real':
                ax[counter].plot(xs, output[name + ' SoC'])
                ax[counter].set_title(name + ' SoC')
                ax[counter].set_ylabel('SoC')
                ax[counter].legend(loc=(1.01, .1))
                counter += 1
        return fig

    def plotReqImpTotalEnergy(self, select='real'):
        if select == 'real':
            output = self.output_real
            ix = 0
        else:
            output = self.output_reactive
            ix = 1
        n_rows = 1 + len(self.resource_names)
        fig, ax = plt.subplots(nrows=n_rows, sharex=True, figsize=(n_rows * 3, 10))
        xs = range(1, len(self.pcc_signal) + 1)
        ax[0].plot(xs, self.tstep * np.cumsum(output['PCC req']), label='requested')
        ax[0].plot(xs, self.tstep * np.cumsum(output['PCC imp']), label='implemented')
        ax[0].plot(xs, self.tstep * np.cumsum(self.pcc_signal[ix, :]), ls='--', label='set point')
        ax[0].set_title('total aggregate energy')
        if select == 'real':
            ax[0].set_ylabel('kWh')
        else:
            ax[0].set_ylabel('kvar-h')
        ax[0].legend(loc=(1.01, .1))
        counter = 1
        for resource in self.resource_list:
            name = resource.name
            key1 = name + ' req'
            key2 = name + ' imp'
            ax[counter].plot(xs, self.tstep * np.cumsum(output[key1]), label='requested')
            ax[counter].plot(xs, self.tstep * np.cumsum(output[key2]), label='implemented')
            ax[counter].set_title(name + ' total energy')
            if select == 'real':
                ax[counter].set_ylabel('kWh')
            else:
                ax[counter].set_ylabel('kvar-h')
            ax[counter].legend(loc=(1.01, .1))
            counter += 1
        return fig


if __name__ == '__main__':
    from resources import *

    # define constants
    mu = 10000
    agg_point = np.concatenate((np.zeros(10),10*np.ones(30),-6*np.arange(10)+10,-50*np.ones(15),15*np.ones(7),np.zeros(8),10*np.ones(20),-10*np.ones(99)))
    T = len(agg_point)
    agg_point_q = .5*agg_point
    agg_point = np.vstack((agg_point,agg_point_q))


    # define resources
    pv1 = PVSysR2('pv1', Cpv=.1)
    batt1 = BatteryR2('batt1', initial_SoC=0.5,target_SoC=0.2, Cb=0)
    disc1 = DiscreteR2('disc1', points=np.array([[-10,-5], [-20, -10], [-30, -15], [-13, -2], [0, 0]]), Cdisc=0)
    disc2 = DiscreteR2('disc2', points=np.array([[-10,-5], [-20, -10], [-30, -15], [-13, -2], [0, 0]]), Cdisc=0)
    disc3 = DiscreteR2('disc3', points=np.array([[-10,-5], [-20, -10], [-30, -15], [-13, -2], [0, 0]]), Cdisc=0)

    # make controller
    contr = ControllerR2(mu=mu)

    # add resources
    #contr.addResource(pv1)
    #contr.addResource(batt1)
    contr.addResource(disc1)
    #contr.addResource(disc2)
    #contr.addResource(disc3)

    print('resource names: ',contr.resource_names)
    print('total time horizon: ', T)

    start_time = time.time()

    """
    for t in range(T):
        # solve optimization for a single step
        p_conv, eps_conv, opt_val = contr.solveStep(agg_point[:,t])

        # get projections onto feasible set
        p_operating = contr.getProjectionsWithError(p_conv)

        # update error term
        contr.updateError(p_conv,p_operating)

        # place data into arrays
        p_conv_all[:,t] = p_conv.flatten()
        p_op_all[:,t] = p_operating.flatten()
        eps_conv_all[:,t] = eps_conv
    """
    #agg_point = agg_point[:,0:2]
    #agg_point = np.array([[-15,-7.5], [-13, -9], [-30, -20]]).T

    dim, T = agg_point.shape
    output = contr.runSimulation(agg_point, solver='MOSEK')

    print('total comp time: ',time.time() - start_time)

    realP = np.array([output['PCC imp'][i] for i in range(T)])

    plt.figure()
    plt.plot(realP[:,0])
    plt.plot(agg_point[0,:])
    plt.figure()
    plt.plot(realP[:,1])
    plt.plot(agg_point[1,:])
    plt.show()

