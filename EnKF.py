# Based on the filterpy EnKF
# https://filterpy.readthedocs.io/en/latest/_modules/filterpy/kalman/ensemble_kalman_filter.html#EnsembleKalmanFilter.update
# Book: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from copy import deepcopy
import numpy as np
from numpy import array, zeros, eye, dot
from numpy.random import multivariate_normal
from filterpy.common import outer_product_sum


class EnKF(object):
    """EnKF for Python synchronous machine model"""
    def __init__(self, x,y, P, dim_z, dt, N, fx, last_time_instant, SM_params, EX_params, GO_params):

        # initialization of ensemble kalman filter

        dim_x = len(x)
        self.dim_x = dim_x  # equals to total state number
        self.dim_z = dim_z  # equals to 2 (P,Q)
        self.dim_y = len(y)  # equals to total algebraic variable number
        self.dt = dt  # delta time
        self.N = N  # number of ensemble
        self.fx = fx  # state transition function, note that this function is only for the states. 
                        # the transition function for the parameters are identity.
        self.K = zeros((dim_x, dim_z))
        self.z = array([[None] * self.dim_z]).T
        self.S = zeros((dim_z, dim_z))  # system uncertainty
        self.SI = zeros((dim_z, dim_z))  # inverse system uncertainty

        self.initialize(x,y, P)

        self.Q = eye(dim_x) * 0.000001  # process uncertainty
        self.R = eye(dim_z) * 0.0001  # measurement uncertainty 
        self.inv = np.linalg.inv

        self._mean = zeros(dim_x)
        self._mean_z = zeros(dim_z)

        self.last_time_instant = last_time_instant  # last time instant
        self.SM_params = SM_params  # synchronous machine parameters
        self.EX_params = EX_params  # exciter parameters
        self.GO_params = GO_params  # governor parameters

    def initialize(self, x,y, P):

        if x.ndim != 1:
            raise ValueError('x must be a 1D array')

        self.sigmas = multivariate_normal(mean=x, cov=P, size=self.N)  # create sigma values
                                        # of mean x, covariance P and size (state_number x N)

        p_y = 0.0001
        self.sigmas_y = multivariate_normal(mean=y, cov=np.identity(self.dim_y)*p_y, size=self.N)  # create sigma values
                                        # of mean y, covariance 0 and size (algebraic_variable_number x N)
        self.x = x
        self.P = P
        self.y = y

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(self, V, theta, u):
        """ Predict next position. """
        # Transition k --> k+1

        N = self.N
        self.sigmas_h = zeros((N, self.dim_z))  # z

        # The sigma values are transitioned into the next time instant
        # with the help of WPE_fx function. This is done sequentially for every sigma (every state vector).
        for i, s in enumerate(self.sigmas):
            self.sigmas_h[i], self.sigmas[i], self.sigmas_y[i] = self.fx(V, theta, self.sigmas[i], self.sigmas_y[i], u,
                                                        self.SM_params, self.EX_params, self.GO_params, self.dt)

        e = multivariate_normal(self._mean, self.Q, N)
        self.sigmas += e

        self.P = outer_product_sum(self.sigmas - self.x) / (N - 1) #+1

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(self, z, R=None):

        """ Update according to the measurements (if any) """
        # The sigma values are updated by x_k+1 (+)  = x_k+1 (-) + K*(z_k+1-h(x_k+1)).

        if z is None:
            self.z = array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if R is None:
            R = self.R
            self.R[0,0] = 0.00001 # P_meas uncertainty
            self.R[1,1] = 0.00001 #Q_meas uncertainty
        if np.isscalar(R):
            R = eye(self.dim_z) * R

        N = self.N
        sigmas_h = self.sigmas_h

        # Take the mean or median of the output predictions.
        z_mean = np.mean(sigmas_h, axis=0)
        # z_mean = np.nanmedian(sigmas_h, axis=0)

        P_zz = (outer_product_sum(sigmas_h - z_mean) / (N - 1)) + R #+1
        P_xz = outer_product_sum(
            self.sigmas - self.x, sigmas_h - z_mean) / (N - 1) #+1

        self.S = P_zz
        self.SI = self.inv(self.S)
        self.K = dot(P_xz, self.SI)

        e_r = multivariate_normal(self._mean_z, R, N)
        for i in range(N):
            self.sigmas[i] += dot(self.K, z + e_r[i] - sigmas_h[i])

        # Take the mean or median of the states.
        self.x = np.mean(self.sigmas, axis=0)
        #self.x = np.nanmedian(self.sigmas, axis=0)
        self.y = np.mean(self.sigmas_y, axis=0)
        #self.y = np.nanmedian(self.sigmas_y, axis=0)

        # Update the last time instant
        self.last_time_instant += self.dt

        self.P = self.P - dot(dot(self.K, self.S), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

