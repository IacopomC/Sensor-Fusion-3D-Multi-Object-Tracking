# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# contains the general parameters for KF implementation
import misc.params as params


class Filter:

    """
    The Kalman Filter Object that implements the homonymous algorithm

    Parameters:

    arg (str): The arg is used for

    Attributes:
    arg (str): This is where we store arg,
    """

    def __init__(self):
        pass

    def F(self):
        """"
        Implement the state transition matrix F for a constant velocity motion model

        Parameters:
        None

        Returns:
        system_matrix (numpy matrix): state transition matrix F for state [x, y, z, vx, vy, vz]
        """

        system_matrix = np.eye(params.dim_state)
        system_matrix[0, 3] = params.dt
        system_matrix[1, 4] = params.dt
        system_matrix[2, 5] = params.dt

        return np.matrix(system_matrix)

    def Q(self):
        """"
        Implement the process noise covariance matrix Q for a constant velocity motion model

        Parameters:
        None

        Returns:
        covariance_matrix (numpy matrix): process noise matrix Q
        """

        covariance_matrix = np.zeros((params.dim_state, params.dim_state))
        covariance_matrix[0, 0] = covariance_matrix[1, 1] = covariance_matrix[2, 2] = (params.dt ** 4) / 4
        covariance_matrix[0, 3] = covariance_matrix[1, 4] = covariance_matrix[2, 5] = (params.dt ** 3) / 2
        covariance_matrix[3, 0] = covariance_matrix[4, 1] = covariance_matrix[5, 2] = (params.dt ** 3) / 2
        covariance_matrix[3, 3] = covariance_matrix[4, 4] = covariance_matrix[5, 5] = (params.dt ** 2)

        covariance_matrix = covariance_matrix * params.q

        return np.matrix(covariance_matrix)

    def predict(self, track):
        """"
        Predict state and covariance matrix for new step

        Parameters:
        track (Track): the current track whose x and P we update

        Returns:
        None
        """

        F = self.F()
        F_t = F.transpose()
        Q = self.Q()

        x = F * track.x  # state prediction
        P = F * track.P * F_t + Q  # update covariance matrix

        track.set_x(x)
        track.set_P(P)

    def update(self, track, meas):
        """"
        Update current belief (state x and covariance P) with associated measurement

        Parameters:
        track (Track): the current track whose x and P we update
        meas ():

        Returns:
        None
        """
        # define measurement function (jacobian)
        H = meas.sensor.get_H(track.x)

        # define residual gamma
        gamma = self.gamma(track, meas)

        # define covariance of residual
        S = self.S(track, meas, meas.sensor.get_H(track.x))
        # calculate Kalman Gain
        K = track.P * H.transpose() * S.I

        # update current state using measurement information
        x = track.x + (K * gamma)

        # update covariance matrix
        I = np.eye(params.dim_state)
        P = (I - K * H) * track.P

        track.set_P(P)
        track.set_x(x)
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        """"
        Calculate residual as difference between measurement and expected measurement based on prediction

        Parameters:
        track (Track): the current track whose x and P we update
        meas (Measurement):

        Returns:
        S (numpy matrix): covariance of residual S
        """

        return meas.z - meas.sensor.get_hx(track.x)  # residual

    def S(self, track, meas, H):
        """"
        Calculate covariance of residual S by mapping the state prediction covariance into measurement space

        Parameters:
        track (Track): the current track whose x and P we update
        meas (Measurement):
        H ():

        Returns:
        S (numpy matrix): covariance of residual S
        """

        return H * track.P * H.transpose() + meas.R  # covariance of residual
