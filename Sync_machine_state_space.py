# CLASSICAL POWER PLANT STATE SPACE MODEL

import numpy as np
from model_functions import network_solver, model_init2, fx
from model_functions import plotter
import matplotlib.pyplot as plt
import pandas as pd

# Synchoronous machine model parameters (Kundur's model)
Tdo = 8.0
Tqo = 1.0
Xd = 1.81
Xd1 = 0.3
Xq = 1.76
Xq1 = 0.65
H = 2.5
D = 0
SM_params = [Tdo, Tqo, Xd, Xd1, Xq, Xq1, H, D]

# IEEE DC1A Exciter model parameters
TE = 1.15
KE = 0 # will be calculated later
TF = 0.62
KF = 0.058
TA = 0.89
KA = 187
A = 0.014
B = 1.55
EX_params = [TE, KE, TF, KF, TA, KA, A, B]

# IEEE TGOV1 Turbine Governor model parameters
TCH = 0.3
TSV = 0.5
RD = 0.05
GO_params = [TCH, TSV, RD]

# Time Simulation parameters
N_step = 20000
dt = 0.001

# Initial conditions for the synchronous machine
Pt = 0.8
Qt = 0.3
V0 = 1.0

V = np.ones(N_step)*V0

# Fault scenario at t=0.1s, duration is 0.05s, 70% reduction in V
V[500:600] = 0.3

theta = np.zeros(N_step)

# Put the initial condition for x and calculate KE.
# Initialize delta and Efd
# delta = 70*np.pi/180    # degrees
# Efd = 1.3               # pu

#x0, y, u, KE = model_init(V[0], theta[0], delta, Efd, SM_params, EX_params)

x0, y, u, KE = model_init2(Pt, Qt, V0, SM_params, EX_params)
EX_params[1] = KE
print("Pe: ", y[4], "-- Qe: ", y[5])

# Then proceed with the simulation
x = x0
x_hist = np.zeros((N_step, len(x0)))
y_hist = np.zeros((N_step, len(y)))
time_vector = np.linspace(0, N_step*dt, N_step)

for i in range(N_step):
    Vnow = V[i]
    thetanow = theta[i]
    y_hist[i, :] = y
    x_hist[i, :] = x
    # Check whether the exciter and governor are disabled or not (in the functions).
    x = fx(x, u, y, SM_params, EX_params, GO_params, dt)
    y = network_solver(Vnow, thetanow, x, y, SM_params)

# Plot the results
Pe = y_hist[:, 4]
Qe = y_hist[:, 5]


fig = plt.figure()
sub1 = fig.add_subplot(211)
sub2 = fig.add_subplot(212)

sub1.plot(time_vector, Pe)
sub1.set_title('Electrical Power (pu)')

sub2.plot(time_vector, Qe)
sub2.set_title('Reactive Power (pu)')
sub2.set_xlabel('Time (s)')

fig.tight_layout()
fig.savefig('sync_machine_state_space.png')

# plot state variable to check
plotter(x_hist, 3, 'asdw', time_vector)

time_vector = time_vector.reshape((1, len(time_vector)))
y_hist_new = np.concatenate((time_vector.T,y_hist), axis=1)
# Save the results to a csv file
columnsy = ['time','Vd', 'Vq', 'Id', 'Iq', 'Pe', 'Qe']
columnsx = ['Eq1', 'Ed1', 'delta', 'w', 'Efd', 'VF', 'VR', 'TM', 'Psv']
dfy = pd.DataFrame(data=y_hist_new, columns=columnsy)
dfy.to_csv('sync_machine_state_space_outputs.csv', index=False)
dfx = pd.DataFrame(data=x_hist, columns=columnsx)
dfx.to_csv('sync_machine_state_space_states.csv', index=False)
