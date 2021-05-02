import os

# from memory_profiler import profile
# # import cProfile, pstats, io
# from line_profiler import LineProfiler
# import kernprof

import yaml
import numpy as np
# import padasip as pa
import statsmodels.api as sm
import time
import logging


logging.basicConfig(level=logging.WARNING, format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')


class RAKF1D:

    def __init__(self, initial_state, system_model, system_model_error,
                 measurement_error, state_error_variance, residual_threshold, adaptive_threshold,
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
            logging.critical(e)
            exit(-1)

    def run(self, measurement_now, timestamp_ms=0, velocity=0, acceleration=0):
        """Runs RAKF 1D algorithm // 1D because the input is scalar
        and seperately seperately we are x,y and Z values to the algorithm

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
            # equation 29  or equation 33 in report
            self.state_model_prediction = (self.system_model * self.state_model) + (velocity * timedelta) + (0.5 * acceleration * (timedelta ** 2))

            # equation 30 or equation 34 in report
            self.state_error_variance_prediction = (
                                                               self.system_model * self.state_error_variance * self.system_model) + self.system_model_error
            # ----------------  Updating  ---------------------------------------

            # equation 35 or equation 38 in report
            self.measurement_prediction = self.state_measurement_relation * self.state_model_prediction

            # equation 34 or equation 37 in report
            self.residual_measurement = measurement_now - self.measurement_prediction

            # equation 33, not  used here as comparison of c is done using rki only
            # for better understanding
            self.residual_measurement_dash = abs(self.residual_measurement / self.measurement_standard_deviation)

            # equation 31 & 32 or equation 35 in report
            if self.residual_measurement_dash <= self.residual_threshold:
                self.residual_weight = 1 / self.measurement_standard_deviation
            else:
                # self.residual_weight = self.residual_threshold / (self.residual_measurement_dash * self.measurement_standard_deviation)
                self.residual_weight = (self.residual_threshold / (self.residual_measurement_dash * 100)) * (
                            1 / self.measurement_standard_deviation)

            # equation 37
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

            # wls_model = sm.WLS(self.measurement_buffer, self.observation_buffer, self.residual_weight_buffer).fit()
            # # self.state_estimation = wls_model.predict(self.observation_buffer[self.estimator_parameter_count - 1])

            wls_model = sm.WLS(self.measurement_buffer, X, self.residual_weight_buffer).fit()
            self.state_estimation = wls_model.predict([X[-1, 0], X[-1, 1], X[-1, 2]])

            # state_estimation = (wls_model.params[0] * X[-1, 0]) + (wls_model.params[1] * X[-1, 1] * timedelta) + (
            #             wls_model.params[2] * X[-1, 2] * (timedelta ** 2) * 0.5)

            endTime = time.time()

            # equation 36 or equation 39 in report
            self.delta_state_estimate = (
                                                self.state_estimation - self.state_model_prediction) / self.state_error_variance_prediction

            # equation 38 or equation 42 in report
            # if self.delta_state_estimate <= self.adaptive_threshold:
            #     self.adaptive_factor = 1
            # else:
            #     self.adaptive_factor = self.adaptive_threshold / \
            #         (self.delta_state_estimate ** 2)

            if self.delta_state_estimate < self.adaptive_threshold:
                self.adaptive_factor = 1.0
            elif self.adaptive_threshold < self.delta_state_estimate < self.residual_threshold:
                self.adaptive_factor = (self.adaptive_threshold / self.delta_state_estimate * 50) # here gama=50
            else:
                self.adaptive_factor = self.delta_state_estimate * 50 # here gama=50 and tunned using hit and trial method

            # equation 39 or equation
            reciprocal_adaptive_factor = 1 / self.adaptive_factor
            reciprocal_residual_weight = 1 / self.residual_weight  # here Pk is residual weight
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
            logging.critical(e)
            exit(-1)


#
# w = fc / (fs / 2) # Normalize the frequency
# b, a = signal.butter(5, w, 'low')
# output = signal.filtfilt(b, a, signalc)

class RAKF2D:
    """Class implementation for Robust Adaptive Kalman Filter for 2 Dimension
    """

    def __init__(self, file_name, tag_id):
        """Initializes RAKF 2D class object

        Args:
            file_name (with extension .yaml): configuration yaml file containing configuration data for RAKF with corresponding Tag ID
            tag_id (int): Configuration data for corresponding Tag ID in configuration file to be used
        """
        try:
            self.rakf_x = None
            self.rakf_y = None
            self.pos_x = 0
            self.pos_y = 0
            self.vel_x = 0
            self.vel_y = 0
            self.time_previous = -1.0

            model_error = []
            measurement_error = []
            residual_threshold = []
            adaptive_threshold = []
            initial_position = []
            estimator_parameter_count = 0

            if os.path.exists(file_name):
                with open(file_name, 'r') as json_file:
                    # client_json = json.load( json_file )
                    client_json = yaml.load(json_file, Loader=yaml.FullLoader)
                    for entry in client_json["rakf_config"]:
                        if entry["tag_id"] == tag_id:
                            # --- add more attributes ------
                            for item in entry["model_error"]:
                                model_error.append(item)
                            for item in entry["measurement_error"]:
                                measurement_error.append(item)
                            for item in entry["residual_threshold"]:
                                residual_threshold.append(item)
                            for item in entry["adaptive_threshold"]:
                                adaptive_threshold.append(item)
                            for item in entry["initial_position"]:
                                initial_position.append(item)
                            estimator_parameter_count = entry["estimator_parameter_count"]
                            forgetting_factor = float(
                                entry["forgetting_factor"])
                            if forgetting_factor > 1 or forgetting_factor < 0:
                                raise ValueError(
                                    "Forgetting factor must be between 0 and 1 (both inclusive)")
                        else:
                            raise ValueError(
                                "No tag with id" + tag_id + "found")
            else:
                raise FileNotFoundError(
                    "File not found, check name of the file or path")

            # Creating an object for all axis (X, Y, Z)
            self.rakf_x = RAKF1D(initial_state=initial_position[0], system_model=1, system_model_error=model_error[0],
                                 measurement_error=measurement_error[0],
                                 state_error_variance=1, residual_threshold=residual_threshold[0],
                                 adaptive_threshold=adaptive_threshold[0],
                                 estimator_parameter_count=estimator_parameter_count,
                                 forgetting_factor=forgetting_factor)

            self.rakf_y = RAKF1D(initial_state=initial_position[1], system_model=1, system_model_error=model_error[1],
                                 measurement_error=measurement_error[1],
                                 state_error_variance=1, residual_threshold=residual_threshold[1],
                                 adaptive_threshold=adaptive_threshold[1],
                                 estimator_parameter_count=estimator_parameter_count,
                                 forgetting_factor=forgetting_factor)

        except Exception as e:
            logging.critical(e)
            exit(-1)

    def run_on_measurements(self, measurement_x, measurement_y, measurement_z, timestamp_ms=0):
        try:
            # run RAKF algorithm over measurements
            res_eqn_x, res_var_x = self.rakf_x.run(
                measurement_now=measurement_x, timestamp_ms=timestamp_ms, velocity=0)
            res_eqn_y, res_var_y = self.rakf_y.run(
                measurement_now=measurement_y, timestamp_ms=timestamp_ms, velocity=0)

            # Get timedelta based on timestamp and update velocity
            if self.time_previous < 0:
                timedelta = 0.0
                self.vel_x = 0.0
                self.vel_y = 0.0
            else:
                timedelta = timestamp_ms - self.time_previous
                self.vel_x = (
                        (self.pos_x - res_var_x["state_model"]) / timedelta)
                self.vel_y = (
                        (self.pos_y - res_var_y["state_model"]) / timedelta)

            self.time_previous = timestamp_ms

            # update current position
            self.pos_x = res_var_x["state_model"]
            self.pos_y = res_var_y["state_model"]

            return [res_var_x, res_var_y]
        except Exception as e:
            logging.critical(e)
            exit(-1)


class RAKF3D:
    """Class implementation for Robust Adaptive Kalman Filter for 3 Dimension
    """

    def __init__(self, file_name, tracker_id):
        """Initializes RAKF 3D class object

        Args:
            file_name (with extension .yaml): configuration yaml file containing configuration data for RAKF with corresponding Tag ID
            tag_id (int): Configuration data for corresponding Tag ID in configuration file to be used
        """
        try:
            self.rakf_x = None
            self.rakf_y = None
            self.rakf_z = None
            self.pos_x = 0
            self.pos_y = 0
            self.pos_z = 0
            self.vel_x = 0
            self.vel_y = 0
            self.vel_z = 0
            self.time_previous = -1.0

            model_error = []
            measurement_error = []
            residual_threshold = []
            adaptive_threshold = []
            initial_position = []
            estimator_parameter_count = 0

            if os.path.exists(file_name):
                with open(file_name, 'r') as json_file:
                    # client_json = json.load( json_file )
                    client_json = yaml.load(json_file, Loader=yaml.FullLoader)
                    for entry in client_json["tracker_config"]:
                        if entry["tracker_id"] == tracker_id:
                            # --- add more attributes ------
                            for item in entry["model_error"]:
                                model_error.append(item)
                            for item in entry["measurement_error"]:
                                measurement_error.append(item)
                            for item in entry["residual_threshold"]:
                                residual_threshold.append(item)
                            for item in entry["adaptive_threshold"]:
                                adaptive_threshold.append(item)
                            for item in entry["initial_position"]:
                                initial_position.append(item)
                            estimator_parameter_count = entry["estimator_parameter_count"]
                            forgetting_factor = float(
                                entry["forgetting_factor"])
                            if forgetting_factor > 1 or forgetting_factor < 0:
                                raise ValueError(
                                    "Forgetting factor must be between 0 and 1 (both inclusive)")
                        else:
                            raise ValueError(
                                "No tag with id" + tracker_id + "found")
            else:
                raise FileNotFoundError(
                    "File not found, check name of the file or path")

            # Creating an object for all axis (X, Y, Z)

            # from pstats import SortKey
            # profiler = cProfile.Profile()
            # profiler.enable()
            self.rakf_x = RAKF1D(initial_state=initial_position[0], system_model=1, system_model_error=model_error[0],
                                 measurement_error=measurement_error[0],
                                 state_error_variance=1, residual_threshold=residual_threshold[0],
                                 adaptive_threshold=adaptive_threshold[0],
                                 estimator_parameter_count=estimator_parameter_count,
                                 forgetting_factor=forgetting_factor)
            # profiler.disable()
            # stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
            # stats.print_stats()

            self.rakf_y = RAKF1D(initial_state=initial_position[1], system_model=1, system_model_error=model_error[1],
                                 measurement_error=measurement_error[1],
                                 state_error_variance=1, residual_threshold=residual_threshold[1],
                                 adaptive_threshold=adaptive_threshold[1],
                                 estimator_parameter_count=estimator_parameter_count,
                                 forgetting_factor=forgetting_factor)

            self.rakf_z = RAKF1D(initial_state=initial_position[2], system_model=1, system_model_error=model_error[2],
                                 measurement_error=measurement_error[2],
                                 state_error_variance=1, residual_threshold=residual_threshold[2],
                                 adaptive_threshold=adaptive_threshold[2],
                                 estimator_parameter_count=estimator_parameter_count,
                                 forgetting_factor=forgetting_factor)
        except Exception as e:
            logging.critical(e)
            exit(-1)

    def run_on_measurements(self, data):
        try:
            timestamp_ms = data["timestamp_ms"]

            # from pstats import SortKey
            # profiler = cProfile.Profile()
            # profiler.enable()

            # run RAKF algorithm over measurements
            res_eqn_x, res_var_x = self.rakf_x.run(
                measurement_now=data["measurement_x"], timestamp_ms=timestamp_ms, velocity=data["velocity_x"],
                acceleration=data["acceleration_x"])
            # profiler.disable()
            # stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
            # stats.print_stats()

            res_eqn_y, res_var_y = self.rakf_y.run(
                measurement_now=data["measurement_y"], timestamp_ms=timestamp_ms, velocity=data["velocity_y"],
                acceleration=data["acceleration_y"])
            res_eqn_z, res_var_z = self.rakf_z.run(
                measurement_now=data["measurement_z"], timestamp_ms=timestamp_ms, velocity=data["velocity_z"],
                acceleration=data["acceleration_z"])

            # Get timedelta based on timestamp and update velocity
            if self.time_previous < 0:
                timedelta = 0.0
                self.vel_x = 0.0
                self.vel_y = 0.0
                self.vel_z = 0.0
            else:
                timedelta = timestamp_ms - self.time_previous
                self.vel_x = (
                        (self.pos_x - res_var_x["state_model"]) / timedelta)
                self.vel_y = (
                        (self.pos_y - res_var_y["state_model"]) / timedelta)
                self.vel_z = (
                        (self.pos_z - res_var_z["state_model"]) / timedelta)

            self.time_previous = timestamp_ms

            # update current position
            self.pos_x = res_var_x["state_model"]
            self.pos_y = res_var_y["state_model"]
            self.pos_z = res_var_z["state_model"]

            return [res_var_x, res_var_y, res_var_z, res_eqn_x, res_eqn_y, res_eqn_z]
        except Exception as e:
            logging.critical(e)
            exit(-1)
