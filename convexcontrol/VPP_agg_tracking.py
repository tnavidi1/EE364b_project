# VPP aggregate tracking test case

import numpy as np

from resources import *
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from main import ControllerR2

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
        batteries = np.arange(self.N)[[isinstance(r, BatteryR2) for r in  self.resource_list]]
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
            output_reactive.loc[t]['eps'] = self.eps[1]
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
        self.eps = np.squeeze(np.asarray(eps.value))
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

    def plotReqImpPower(self, select='real', show_tcl_desired=True):
        batteries = np.arange(self.N)[[isinstance(r, BatteryR2) for r in self.resource_list]]
        if select == 'real':
            output = self.output_real
            ix = 0
            n_rows = 1 + len(self.resource_names) + len(batteries)
        else:
            output = self.output_reactive
            ix = 1
            n_rows = 1 + len(self.resource_names)
        fig, ax = plt.subplots(nrows=n_rows, sharex=True, figsize=(8, n_rows * 2))
        xs = range(1, self.pcc_signal.shape[1] + 1)
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
            is_pv = isinstance(resource, PVSysR2)
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
            if is_tcl and show_tcl_desired:
                p_con = resource.points[resource.desired[:self.pcc_signal.shape[1]], :].T
                ax[counter].plot(xs, p_con[ix], ls='--', label='desired')
            if is_pv and select == 'real':
                ax[counter].plot(xs, resource.power_signal[1:len(xs)+1], ls='--', label='desired')
            ax[counter].legend(loc=(1.01, .1))
            counter += 1
            if is_battery and select == 'real':
                ax[counter].plot(xs, output[name + ' SoC'], label='SoC')
                ax[counter].plot(xs, resource.target_SoC[:len(xs)], label='target SoC')
                ax[counter].set_title(name + ' SoC')
                ax[counter].set_ylabel('SoC')
                ax[counter].legend(loc=(1.01, .1))
                counter += 1
        ax[-1].set_xlabel('time step')
        return fig

    def plotReqImpTotalEnergy(self, select='real'):
        if select == 'real':
            output = self.output_real
            ix = 0
        else:
            output = self.output_reactive
            ix = 1
        n_rows = 1 + len(self.resource_names)
        fig, ax = plt.subplots(nrows=n_rows, sharex=True, figsize=(8, n_rows * 2))
        xs = range(1, self.pcc_signal.shape[1] + 1)
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
        ax[-1].set_xlabel('time step')
        return fig

def convert_TCL_con(solar_data):    
    horizon = len(solar_data)     
    T_ = horizon // 2
    assert T_ == 288 # 5min interval 
    # first-day 
    solar_day1_cum = solar_data[0:T_].cumsum()
    solar_day1_mean = (solar_day1_cum).mean()
    # 
    TCL_day1_points = np.where( (solar_day1_cum < 1.01*solar_day1_mean) & (solar_day1_cum > 0.01*solar_day1_mean), 
                          1, solar_day1_cum)
    
    TCL_day1_points = np.where((TCL_day1_points < solar_day1_cum.max()) & (TCL_day1_points >= 1.0*solar_day1_mean), 
                          2, TCL_day1_points)
    TCL_day1_points = np.where((TCL_day1_points < 1) | (TCL_day1_points >= 1.0*solar_day1_mean),
                               0, TCL_day1_points)
    TCL_day1_points = TCL_day1_points.astype(np.int)
    # second day
    solar_day2_cum = solar_data[T_:].cumsum()
    solar_day2_mean = (solar_day2_cum).mean()
    
    TCL_day2_points = np.where((solar_day2_cum < 0.9*solar_day2_mean), 0, solar_day2_cum)
    TCL_day2_points = np.where((TCL_day2_points < solar_day2_cum.max()) & (TCL_day2_points >= 0.9*solar_day2_mean), 1, TCL_day2_points) 
    TCL_day2_points = np.where((TCL_day2_points >= solar_day2_mean), 0, TCL_day2_points)
    TCL_day2_points = TCL_day2_points.astype(np.int) 
    # 
    TCL_points = np.append(TCL_day1_points, TCL_day2_points)    
    # mask 0-1  
    TCL_points_mask = np.tile(np.concatenate((np.zeros(10), np.ones(10))), 576//20)
    TCL_points_mask = np.concatenate((TCL_points_mask, np.zeros(16)))
    TCL_points_w_duty = TCL_points_mask * TCL_points
    TCL_points_w_duty = np.array(TCL_points_w_duty,dtype=int)   
    
    return TCL_points_w_duty


# load data
Data = np.load('VPP_data.npz')
real_agg=Data['real_agg']
imag_agg=Data['imag_agg']
SOCmax=Data['SOCmax']
battmin=Data['battmin']
battmax=Data['battmax']
PV_inv_max=Data['PV_inv_max']
Q=Data['Q']
Data = np.load('other.npz')
solar = Data['solar']

solar = solar.flatten()

TCL_points_w_duty = convert_TCL_con(solar)

Q = Q.flatten()/12

Q = Q/SOCmax

agg_point = np.vstack((real_agg,imag_agg))

solar = np.r_[solar, 0]

#plt.figure()
#plt.plot(real_agg)
#plt.figure()
#plt.plot(solar)
#plt.show()
solar += 1e-7

# define constants
mu = 100

# define resources
pv1 = PVSysR2('pv1', Cpv=10, data=solar, pmax=PV_inv_max)
batt1 = BatteryR2('batt1', Cb=10, Cbl=0, pmin=battmin, pmax=battmax, initial_SoC=0., target_SoC=Q, capacity=SOCmax, eff=.95,
                 tstep=5./60)

points=np.array([[0, 0], [-.7,-.1], [-1.5, -.5], [-2, -1]])
#points=np.array([[0, 0], [-10,-5], [-20, -10]])
disc1 = DiscreteR2('disc1', points=points, desired=TCL_points_w_duty, Cdisc=100, t_lock=2)

# make controller
contr = ControllerR2(mu=mu)

# add resources
contr.addResource(pv1)
contr.addResource(batt1)
contr.addResource(disc1)

dim, T = agg_point.shape

print('resource names: ',contr.resource_names)
print('total time horizon: ', T)

start_time = time.time()

output = contr.runSimulation(agg_point, solver='MOSEK')

print('total comp time: ',time.time() - start_time)

sns.set(context='talk', style='darkgrid', palette='colorblind')

contr.plotReqImpTotalEnergy(select='real')
contr.plotReqImpTotalEnergy(select='reactive')
contr.plotReqImpPower(select='real')
contr.plotReqImpPower(select='reactive')
plt.tight_layout()
plt.show()




