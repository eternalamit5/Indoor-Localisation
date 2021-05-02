import numpy as np
from Algorithm.ParameterEstimation import ParameterEstimation
import json
import padasip as pa
from scipy import signal
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import time


class RAKF1D:

    def __init__(self, initial_state, system_model, system_model_error,
                 measurement_error, state_error_variance, residual_threshold, adaptive_threshold, region_factors=None,
                 estimator_parameter_count=1, forgetting_factor=0.98, motion_model_type="constant-position"):
        """Initializes RAKF 1 Dimensional instance

        Args:
            initial_state (float): Initial system state
            system_model (float): System model equation coefficient
            system_model_error (float): System model error (variance of model error)
            measurement_error (float): measurement model error (variance of measurement error)
            state_error_variance (float): Initial state error variance
            residual_threshold (float): residual threshold value
            adaptive_threshold (float): Adaptive threshold value
            estimator_parameter_count (int, optional): Sample count for parameter estimation method. Defaults to 1.
            forgetting_factor (float, optional): Forgetting factor for parameter estimation. Defaults to 0.98.
            motion_type (str, optional): Type of motion model. Defaults to "constant-position".
        """
        try:
            self.motion_model_type = motion_model_type

            # timestamp
            self.time_previous = -1.0

            # states
            self.state_model_prediction = None
            self.state_model = initial_state  # X

            # system model
            self.system_model = system_model  # A
            self.system_model_error = system_model_error  # Q

            # measurement
            self.state_measurement_relation = 1  # C
            self.measurement_standard_deviation = np.sqrt(measurement_error)
            self.measurement_prediction = None

            # residual
            self.residual_threshold = residual_threshold  # c
            self.residual_weight = None
            self.residual_measurement = None
            self.residual_measurement_dash = None

            # state error variance
            self.state_error_variance_prediction = None
            self.state_error_variance = state_error_variance  # P

            # state estimation
            self.state_estimation = None
            self.delta_state_estimate = None

            # gain
            self.gain = None

            # adaptive
            self.adaptive_factor = None
            self.adaptive_threshold = adaptive_threshold  # co

            # parameter estimation
            self.measurement_buffer = np.zeros(estimator_parameter_count)
            self.residual_weight_buffer = np.ones(estimator_parameter_count)
            self.observation_buffer = np.zeros(estimator_parameter_count)
            self.param_est = sm.WLS(self.measurement_buffer, self.observation_buffer, self.residual_weight_buffer)
            self.estimator_parameter_count = estimator_parameter_count

            self.velocity_buffer = np.zeros(estimator_parameter_count)
            self.acceleration_buffer = np.zeros(estimator_parameter_count)
            # self.param_est = pa.filters.FilterRLS(estimator_parameter_count, mu=forgetting_factor, w=np.zeros(
            #     estimator_parameter_count))  # mu is forgetting factor
        except Exception as e:
            print("exception RAKF1D__init")
            exit(-1)

    def run(self, measurement_now, timestamp_ms=0, velocity=0, acceleration=0):
        """Runs RAKF 1D algorithm

        Args:
            :param velocity (int, optional): Velocity in meter per second. Defaults to 0.
            :param timestamp_ms (int, optional): Timestamp in milliseconds (since epoch). Defaults to 0.
            :param measurement_now (float):  Measurement
            :param acceleration:
        Returns:
            eqn_result, variable_result : Equation result and Variable result are dictionaries containing results of various parameters used in the algorithm calculation

        """
        try:

            # Get timedelta based on timestamp
            if self.time_previous < 0:
                timedelta = 0.0
            else:
                timedelta = timestamp_ms - self.time_previous

            self.time_previous = timestamp_ms
            # -----------------  Prediction  -----------------------------------
            # equation 29
            self.state_model_prediction = (self.system_model * self.state_model) + (velocity * timedelta) + (
                    acceleration * (timedelta ** 2) * 0.5)

            # equation 30
            self.state_error_variance_prediction = (
                                                           self.system_model * self.state_error_variance * self.system_model) + self.system_model_error
            # ----------------  Updating  ---------------------------------------

            # equation 35
            self.measurement_prediction = self.state_measurement_relation * self.state_model_prediction

            # equation 34
            self.residual_measurement = measurement_now - self.measurement_prediction

            # equation 33
            self.residual_measurement_dash = abs(self.residual_measurement / self.measurement_standard_deviation)

            # equation 31 & 32
            if self.residual_measurement_dash <= self.residual_threshold:
                self.residual_weight = 1 / self.measurement_standard_deviation
            else:
                # self.residual_weight = self.residual_threshold / (self.residual_measurement_dash * self.measurement_standard_deviation)
                self.residual_weight = (self.residual_threshold / (self.residual_measurement_dash * 100)) * (
                        1 / self.measurement_standard_deviation)

            # equation 37
            # np.roll(self.measurement_buffer, 1)
            # self.measurement_buffer[0] = measurement_now
            # self.state_estimation = self.param_est.predict(
            #     self.measurement_buffer)

            startTime = time.time()

            self.observation_buffer = np.roll(self.observation_buffer, -1)  # Observed position
            self.observation_buffer[self.estimator_parameter_count - 1] = self.state_model

            self.velocity_buffer = np.roll(self.velocity_buffer, -1)  # Observed velocity
            self.velocity_buffer[self.estimator_parameter_count - 1] = velocity
            self.acceleration_buffer = np.roll(self.acceleration_buffer, -1)
            self.acceleration_buffer[self.estimator_parameter_count - 1] = acceleration  # Observed acceleration
            X = np.stack([self.observation_buffer, self.velocity_buffer, self.acceleration_buffer], axis=1)

            self.measurement_buffer = np.roll(self.measurement_buffer, -1)
            self.measurement_buffer[self.estimator_parameter_count - 1] = measurement_now  # measurement

            wls_model = sm.WLS(self.measurement_buffer, self.observation_buffer, self.residual_weight_buffer).fit()
            # print(wls_model.params)
            # self.state_estimation = wls_model.predict(self.observation_buffer[self.estimator_parameter_count - 1])

            wls_model = sm.WLS(self.measurement_buffer, X, self.residual_weight_buffer).fit()
            # print(wls_model.params)
            # print(("\n"))
            self.state_estimation = wls_model.predict([X[-1, 0], X[-1, 1], X[-1, 2]])
            state_estimation = (wls_model.params[0] * X[-1, 0]) + (wls_model.params[1] * X[-1, 1] * timedelta) + (
                    wls_model.params[2] * X[-1, 2] * (timedelta ** 2) * 0.5)
            # print(self.state_estimation)
            # print(state_estimation)
            # print("\n")

            endTime = time.time()

            # equation 36
            self.delta_state_estimate = (
                                                self.state_estimation - self.state_model_prediction) / self.state_error_variance_prediction

            # equation 38
            # if self.delta_state_estimate <= self.adaptive_threshold:
            #     self.adaptive_factor = 1
            # else:
            #     self.adaptive_factor = self.adaptive_threshold / \
            #         (self.delta_state_estimate ** 2)

            if self.delta_state_estimate < self.adaptive_threshold:
                self.adaptive_factor = 1.0
            elif self.adaptive_threshold < self.delta_state_estimate < self.residual_threshold:
                self.adaptive_factor = (self.adaptive_threshold / self.delta_state_estimate * 50)
            else:
                self.adaptive_factor = self.delta_state_estimate * 50

            # equation 39
            reciprocal_adaptive_factor = 1 / self.adaptive_factor
            reciprocal_residual_weight = 1 / self.residual_weight
            numerator = reciprocal_adaptive_factor * self.state_error_variance_prediction * self.state_measurement_relation
            denominator = (
                                  reciprocal_adaptive_factor * self.state_measurement_relation * self.state_error_variance_prediction * self.state_measurement_relation) + reciprocal_residual_weight
            self.gain = numerator / denominator

            # equation 40
            self.state_model = self.state_model_prediction + (self.gain * self.residual_measurement)

            # equation 41
            # not done here, as normalization is not need for 1 D

            # equation 42
            self.state_error_variance = (
                                                1 - self.gain * self.state_measurement_relation) * self.state_error_variance_prediction

            # Activity related to eqn 37 , update parameters in parameter estimation based on states
            # self.param_est.adapt(self.state_model, self.measurement_buffer)
            self.residual_weight_buffer = np.roll(self.residual_weight_buffer, -1)
            self.residual_weight_buffer[self.estimator_parameter_count - 1] = self.residual_weight  # Weight

            # print(f"Runtime of the program is {endTime - startTime}")

            eqn_result = {
                "residual_threshold": self.residual_threshold,
                "adaptive_threshold": self.adaptive_threshold,
                "eqn29": self.state_model_prediction,
                "eqn30": self.state_error_variance_prediction,
                "eqn35": self.measurement_prediction,
                "eqn34": self.residual_measurement,
                "eqn33": self.residual_measurement_dash,
                "eqn31": self.residual_weight,
                "eqn37": self.state_estimation,
                "eqn36": self.delta_state_estimate,
                "eqn38": self.adaptive_factor,
                "eqn39": self.gain,
                "eqn39_numerator": numerator,
                "eqn39_denominator": denominator,
                "eqn40": self.state_model,
                "eqn42": self.state_error_variance
            }

            variable_result = {
                "state_model_prediction": self.state_model_prediction,
                "state_error_variance_prediction": self.state_error_variance_prediction,
                "measurement_prediction": self.measurement_prediction,
                "residual_measurement": self.residual_measurement,
                "residual_measurement_dash": self.residual_measurement_dash,
                "residual_threshold": self.residual_threshold,
                "residual_weight": self.residual_weight,
                "state_estimation": self.state_estimation,
                "delta_state_estimate": self.delta_state_estimate,
                "adaptive_threshold": self.adaptive_threshold,
                "adaptive_factor": self.adaptive_factor,
                "gain_numerator": numerator,
                "gain_denominator": denominator,
                "gain": self.gain,
                "state_model": self.state_model,
                "state_error_variance": self.state_error_variance
            }
            return eqn_result, variable_result
        except Exception as e:
            print("exception RAKF1D run")
            exit(-1)

    # def __init__(self, Amat, Bmat, Cmat, U, Gk, P, Q, R, c, c0, region_1, region_2, region_3, stateX, sampleSize=0):
    #     '''
    #
    #     :param Amat: Matrix describing Human motion dynamics
    #     :param Bmat: vector describing Inputs model to the human
    #     :param Cmat: vector describing sensor model
    #     :param U: Input to human
    #     :param Gk: Kalman gain
    #     :param P: Error estimation
    #     :param Q: Model error (variance)
    #     :param R: measurement error (variance)
    #     :param stateX: state variables in the system
    #     :param c:
    #     :param c0:
    #     :param region_1:
    #     :param region_2:
    #     :param region_3:
    #     :param sampleSize:
    #     '''
    #     self.Xhat = stateX
    #     self.Amat = Amat
    #     self.Bmat = Bmat
    #     self.Cmat = Cmat
    #     self.U = U
    #     self.Gk = Gk
    #     self.P = P
    #     self.Q = Q
    #     self.R = R
    #     self.stdDeviationOfR = np.sqrt( R )
    #     self.c = c
    #     self.c0 = c0
    #     self.region_1 = region_1
    #     self.region_2 = region_2
    #     self.region_3 = region_3
    #
    #     self.previous_update_time_ms = 0
    #
    #     # Parameter estimation variables
    #     if sampleSize > 0:
    #         self.parameter_est = ParameterEstimation(sampleSize=sampleSize)
    #     else:
    #         self.parameter_est = None
    #
    # def run(self, Z, timenow_ms):
    #
    #     dt = timenow_ms - self.previous_update_time_ms
    #     self.previous_update_time_ms = timenow_ms
    #
    #     # Prediction
    #     self.Xhat = (self.Amat * self.Xhat)
    #     self.P = self.Amat * (self.P * self.Amat) + self.Q
    #
    #     # Priori values
    #     XhatPrior = self.Xhat  # Variable to store predicted value of Xhat
    #     PPrior = self.P  # Variable to store the predicted value of P
    #
    #     # Robust Calculation
    #     rk = Z - self.Cmat * XhatPrior
    #     rkDash = abs( rk )
    #
    #     if rkDash <= self.c:
    #         Pbar = self.c / self.stdDeviationOfR
    #     elif rkDash > self.c:
    #         Pbar = (self.c / rkDash) * (1 / self.stdDeviationOfR)
    #
    #     # Adaptive calculation
    #     if self.parameter_est is None:
    #         Xtilde = (1 / (self.Cmat * Pbar * self.Cmat)) * (self.Cmat * Pbar * Z)
    #     else:
    #         Xtilde = self.parameter_est.parameter_estimation(Z, Pbar)
    #
    #     delXtilde = abs( Xtilde - self.Xhat ) / np.sqrt( self.P )
    #
    #     if delXtilde <= self.c0:
    #         doh = self.region_1
    #     elif self.c0 < delXtilde < self.c:
    #         doh = self.region_2 * delXtilde
    #     elif delXtilde > self.c:
    #         doh = self.region_3 * delXtilde
    #     else:
    #         doh = 1
    #
    #     # Kalman Gain calculation
    #     innov = (1 / doh) * self.Cmat * self.P * self.Cmat + (1 / Pbar)
    #     self.Gk = ((1 / doh) * (self.P * self.Cmat)) / innov
    #
    #     # Update
    #     self.Xhat = self.Xhat + self.Gk * (Z - (self.Cmat * self.Xhat))  # State update
    #     self.P = (np.array( [1] ) - (self.Gk * self.Cmat)) * self.P  # Variance update
    #
    #     return self.Xhat

#
# w = fc / (fs / 2) # Normalize the frequency
# b, a = signal.butter(5, w, 'low')
# output = signal.filtfilt(b, a, signalc)
