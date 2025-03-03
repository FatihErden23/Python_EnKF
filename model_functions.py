# This file contains the functions that are used to calculate the network equations and 
# the state derivatives for the synchronous machine, exciter, and governor models.
import numpy as np
import matplotlib.pyplot as plt

def network_solver(V, theta, x, y, SM_params):
    # Calculate the network equations and return the algebraic variables
    # x         : State vector
    # y         : Algebraic variables vector
    # SM_params : Synchoronous machine model parameters

    # Unpack the state variables
    delta = x[2]
    Ed = x[1]
    Eq = x[0]

    # Unpack the parameters
    Xd1, Xq1 = SM_params[3], SM_params[5]

    Vd = V * np.sin(delta - theta)
    Vq = V * np.cos(delta - theta)

    Id = (Eq - Vq) / Xd1
    Iq = (Vd-Ed) / Xq1

    Pe = Vd * Id + Vq * Iq
    Qe = Vq * Id - Vd * Iq

    # Pack the algebraic variables into the y vector
    y[0] = Vd
    y[1] = Vq
    y[2] = Id
    y[3] = Iq
    y[4] = Pe
    y[5] = Qe

    return y

def model_init(V, theta, delta, Efd, SM_params, EX_params):
    # Model will be initialized with the given parameters to steady state.
    # V         : Voltage magnitude at POI
    # theta     : Voltage angle at POI
    # delta     : Initial angle of the generator
    # Efd       : Initial excitation voltage
    # SM_params : Synchoronous machine model parameters
    # EX_params : IEEE DC1A Exciter model parameters
    # No need for GO_params.

    # Initialize state, input and algebraic variable vectors
    states = np.zeros(9)
    u = np.zeros(2)
    y = np.zeros(6)

    # Calculate the KE value
    KE = -EX_params[6]*np.exp(EX_params[7]*Efd)

    # First assign the initial conditions for delta (s3) and Efd (s5)
    states[2] = delta
    states[4] = Efd      

    Vd = V*np.sin(delta - theta)
    Vq = V*np.cos(delta - theta)

    # Unpack the reactances
    Xd, Xd1, Xq, Xq1 = SM_params[2], SM_params[3], SM_params[4], SM_params[5]

    Id = (Efd - Vq) / Xd
    Iq = (Vd) / Xq

    Pe = Vd * Id + Vq * Iq
    Qe = Vq * Id - Vd * Iq    

    # Calculate the initial conditions for the rest of the states
    states[0] = Vq + Xd1*Id    # Eq
    states[1] = (Xq-Xq1)*Iq                   # Ed
    states[3] = 1                                 # w        
    states[5] = 0                                 # VF
    states[6] = 0                                 # VR
    states[7] = Pe                                # TM
    states[8] = Pe                                # Psv

    # Calculate the algebraic variables
    y[0] = Vd                                    # Vd
    y[1] = Vq                                    # Vq
    y[2] = Id                                    # Id
    y[3] = Iq                                    # Iq
    y[4] = Pe                                    # Pe
    y[5] = Qe                                    # Qe

    # Calculate and pack the input vector
    u[0] = V                                      # Vref
    u[1] = Pe                                     # Pc

    # Check the initial power values
    print("Initial Pe: ", y[4], "-- Initial Qe: ", y[5])

    return states, y, u, KE

def fx(states, u, y, SM_params, EX_params, GO_params, dt):
    # Calculate the state derivatives and update the state vector for the next time step
    # states : Current state vector
    # u      : Input vector
    # y      : Algebraic variables
    # dt     : Time step

    # Unpack the states
    Eq, Ed, delta, w, Efd, VF, VR, TM, Psv = states

    # Unpack the inputs
    Vref, Pc = u

    # Unpack the algebraic variables
    Vd, Vq, Id, Iq, Pe, Qe = y

    V = np.sqrt(Vd**2 + Vq**2)
    # Unpack the parameters
    Tdo, Tqo, Xd, Xd1, Xq, Xq1, H, D = SM_params
    TE, KE, TF, KF, TA, KA, A, B = EX_params
    TCH, TSV, RD = GO_params

    SE = A*np.exp(B*Efd)

    # Calculate the derivatives
    dEq = (-Eq -(Xd-Xd1)*Id + Efd) / Tdo
    dEd = (-Ed +(Xq-Xq1)*Iq) / Tqo
    dDelta = w - 1                              # ws = 1    
    dW = (TM - Pe - D*(w-1)) / (2*H)
    dEfd = (-(KE+SE)*Efd + VR) / TE
    dVF = (-VF + KF*VR/TE - KF*(KE+SE)*Efd/TE) / TF
    dVR = (-VR + KA*(Vref-VF-V)) / TA
    dTM = (Psv - TM) / TCH
    dPsv = (-Psv + Pc - (w-1)/RD) / TSV

    # Update the states
    # Apply  euler method for the numerical solution
    # x_k+1 = x_k + dt*F(x_k)
    states[0] += dEq * dt
    states[1] += dEd * dt
    states[2] += dDelta * dt
    states[3] += dW * dt

    exc_disable = False
    if exc_disable:
        states[4] = Efd
    else:
        states[4] += dEfd * dt
    
    states[5] += dVF * dt
    states[6] += dVR * dt

    gov_disable = False
    if gov_disable:
        states[7] = TM
        states[8] = Psv
    else:
        states[7] += dTM * dt
        states[8] += dPsv * dt

    return states

def model_init2(Pt, Qt, V, SM_params, EX_params):
    # Model will be initialized with the given machine operation to steady state. REF:Kundur
    # Pt        : Real power at POI
    # Qt        : Reactive power at POI
    # V         : Voltage magnitude at POI
    # SM_params : Synchoronous machine model parameters
    # EX_params : IEEE DC1A Exciter model parameters
    # No need for GO_params.

    # Unpack the synchronous machine parameters
    Xd, Xd1, Xq, Xq1 = SM_params[2], SM_params[3], SM_params[4], SM_params[5]

    # Calculate the terminal current.
    S = np.sqrt(Pt**2 + Qt**2)
    I = S / V
    phi = np.arccos(Pt / S)*np.sign(Qt)

    # Calculate the internal rotor angle
    delta = np.arctan((Xq*I*np.cos(phi))/(V + Xq*I*np.sin(phi)))

    # Calculate the terminal dq components
    Vd = V*np.sin(delta)
    Vq = V*np.cos(delta)
    Id = I*np.sin(phi + delta)
    Iq = I*np.cos(phi + delta)

    # Calculate the internal dq voltages
    Ed = Vd - Xq1*Iq
    Eq = Vq + Xd1*Id

    # Calculate the initial excitation voltage
    Efd = Eq + (Xd - Xd1)*Id

    # Initial mechanical power equals the terminal power
    Pe = Pt
    Tm = Pe

    # Calculate the initial KE value
    KE = -EX_params[6]*np.exp(EX_params[7]*Efd)

    # Initialize the state, input and algebraic variable vectors
    states = np.zeros(9)
    u = np.zeros(2)
    y = np.zeros(6)

    # Assign the initial conditions of the states
    states[0] = Eq
    states[1] = Ed
    states[2] = delta
    states[3] = 1
    states[4] = Efd
    states[5] = 0
    states[6] = 0
    states[7] = Tm
    states[8] = Tm

    # Calculate the algebraic variables
    y[0] = Vd
    y[1] = Vq
    y[2] = Id
    y[3] = Iq
    y[4] = Pe
    y[5] = Qt

    # Calculate the input vector
    u[0] = V
    u[1] = Pe
    
    return states, y, u, KE

def plotter(data_hist,index,name, time_vector):
    # Plot the simulation results
    # data_hist : Simulation results
    # index     : Index of the variable to be plotted
    # name      : Name of the variable to be plotted

    ydata = data_hist[:,index]
    plt.figure()
    plt.plot(time_vector, ydata)
    plt.xlabel('Time (s)')
    plt.ylabel(name)
    plt.title(name + ' vs Time')
    #plt.grid()
    #plt.show()
    plt.savefig(name + '.png')
    return

def parameter_update(x, idx ,SM_params, EX_params, GOV_params):
    # Update the parameters of the synchronous machine model (single parameter case)
    # x         : Parameter vector
    # idx       : Indexes of the parameter and machine submodel to be updated
    # SM_params : Synchoronous machine model parameters
    # EX_params : IEEE DC1A Exciter model parameters
    # GOV_params: IEEE TGOV1 Turbine Governor model parameters

    param = x[-1]

    model, param_idx = idx

    match model:
        case 1:
            SM_params[param_idx] = param
        case 2:
            EX_params[param_idx] = param
        case 3:
            GOV_params[param_idx] = param
    
    return SM_params, EX_params, GOV_params

def enkf_fx(V,theta, x, y, u, SM_params, EX_params, GO_params, dt):
    # Calculate the state derivatives and update the state and observation
    # vector for the next time step

    # V         : Voltage magnitude at POI
    # theta     : Voltage angle at POI
    # x         : State vector
    # y         : Algebraic variables vector
    # u         : Input vector
    # SM_params : Synchoronous machine model parameters
    # EX_params : IEEE DC1A Exciter model parameters
    # GO_params : IEEE TGOV1 Turbine Governor model parameters
    # dt        : Time step

    # Update the model parameter for given ensemble member
    SM_params, EX_params, GO_params = parameter_update(x, (1,6), SM_params, EX_params, GO_params)

    num_param = 1
    # Obtain the model states and calibration parameters from x
    x_s = x[:-1]

    # Calculate the new state prediction
    x_s = fx(x_s, u, y, SM_params, EX_params, GO_params, dt)

    # Calculate the algebraic variables
    y = network_solver(V, theta, x_s, y, SM_params)

    # Obtain the observation vector from y: Pe,Qe
    h = np.array([y[4], y[5]])

    # Append the parameters to the state vector
    idx = 6
    x = np.append(x_s, SM_params[idx])

    return h,x,y
    

