# This is the script for the Ensemble Kalman Filter (EnKF) parameter estimation.

# IMPORT MODULES
import numpy as np
from model_functions import network_solver, model_init2, fx, enkf_fx
from EnKF import EnKF
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

# Playback voltage and angle
V = np.ones(N_step)*V0
# Fault scenario at t=0.1s, duration is 0.05s, 70% reduction in V, balanced fault
V[500:600] = 0.3

theta = np.zeros(N_step)

# Put the initial condition for x and calculate KE.
# Initialize delta and Efd
# delta = 70*np.pi/180    # degrees
# Efd = 1.3               # pu

#x0, y, u, KE = model_init(V[0], theta[0], delta, Efd, SM_params, EX_params)
x0, y, u, KE = model_init2(Pt, Qt, V0, SM_params, EX_params)
EX_params[1] = KE
print("Initially, Pe: ", y[4], "-- Qe: ", y[5])

# Change one of the synchronous machine parameters, this will be calibrated.
SM_params[6] = 3.0

# Initial conditions for the EnKF
p_states = 0.01  # initial state error covariance /0.01
p_param  = 0.01 # initial parameter error covariance /0.01
starting_time = 500 # start the estimation after 500 steps of simulation
N = 100 # number of ensemble members

last_time_instant = starting_time*dt

# Simulate the model until the calibration starting time
x = x0
x_hist = np.zeros((N_step, len(x0)))
y_hist = np.zeros((N_step, len(y)))
p_hist = np.zeros((N_step, 1))

for i in range(starting_time):
    Vnow = V[i]
    thetanow = theta[i]
    y_hist[i, :] = y
    x_hist[i, :] = x
    p_hist[i] = SM_params[6]
    # Check whether the exciter and governor are disabled or not (in the functions).
    x = fx(x, u, y, SM_params, EX_params, GO_params, dt)
    y = network_solver(Vnow, thetanow, x, y, SM_params)

# ENSEMBLE KALMAN FILTER SECTION
# Append the parameter to be estimated to the state vector
x = np.append(x, SM_params[6]) # x now includes the parameter to be estimated
covariance_matrix = np.identity(len(x))*p_states
covariance_matrix[-1, -1] = p_param

# Prepare the measurements.
meas = pd.read_csv("meas.csv")
meas = meas.to_numpy()

t_meas = meas[starting_time:N_step,0]
P_meas = meas[starting_time:N_step,5]
Q_meas = meas[starting_time:N_step,6]

# Construct the EnKF object
enkf = EnKF(x, covariance_matrix, 2, dt, N, enkf_fx, last_time_instant, SM_params, EX_params, GO_params)
z = np.zeros((2))

# Calibration loop
for t in range(starting_time, N_step):
    # get the measurements
    z[0] = P_meas[t+1] # t de olabilir
    z[1] = Q_meas[t+1]
    meas_time = t_meas[t+1]

    # predict the states
    enkf.predict(V[t+1], theta[t+1], u)

    # update the states
    enkf.update(z)

    # Record the state and output vectors.

    # Take the updated parameter value to the history, also print it.
    process_time = t*dt
    p_hist[t] = enkf.x[-1]
    print("Process_t: ", process_time, " -- meas_t:", meas_time, " -- Parameter: ", enkf.x[-1])

print("Calibration process completed.")

# Plotting sequence GELICEK