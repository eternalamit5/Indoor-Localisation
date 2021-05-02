import os
import numpy as np
import yaml

from Algorithm.RAKF1D import RAKF1D
from Utilities.DataQueue import DataQueue
from Communication.TelegraphMQTTClient import telegraph_mqtt_client
import time


class RAKF3D:
    def __init__(self, json_file_name, tag_id):
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
        self.receive_data_queue = DataQueue()

        model_error = []
        measurement_error = []
        residual_threshold = []
        adaptive_threshold = []
        initial_position = []
        estimator_parameter_count = 0

        if os.path.exists(json_file_name):
            with open(json_file_name, 'r') as json_file:
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
                        region_factors = [entry["region_1_factor"], entry["region_2_factor"], entry["region_3_factor"]]
                        forgetting_factor = entry["forgetting_factor"]
                    else:
                        print("No tag with id" + tag_id + "found")
                        exit(-1)

        # RAKF_CLM Creating an object for all axis (X, Y, Z)
        self.rakf_x = RAKF1D(initial_state=initial_position[0], system_model=1, system_model_error=1,  # model_error[0]
                             measurement_error=measurement_error[0],
                             state_error_variance=1, residual_threshold=residual_threshold[0],
                             adaptive_threshold=adaptive_threshold[0],
                             estimator_parameter_count=estimator_parameter_count, forgetting_factor=forgetting_factor,
                             region_factors=region_factors)

        self.rakf_y = RAKF1D(initial_state=initial_position[1], system_model=1, system_model_error=1,  # model_error[1]
                             measurement_error=measurement_error[1],
                             state_error_variance=1, residual_threshold=residual_threshold[1],
                             adaptive_threshold=adaptive_threshold[1],
                             estimator_parameter_count=estimator_parameter_count, forgetting_factor=forgetting_factor,
                             region_factors=region_factors)

        self.rakf_z = RAKF1D(initial_state=initial_position[2], system_model=1, system_model_error=1,  # model_error[2]
                             measurement_error=measurement_error[2],
                             state_error_variance=1, residual_threshold=residual_threshold[2],
                             adaptive_threshold=adaptive_threshold[2],
                             estimator_parameter_count=estimator_parameter_count, forgetting_factor=forgetting_factor,
                             region_factors=region_factors)

        # RAKF_CVM Creating an object for all axis (X, Y, Z)
        self.rakf_cvm_x = RAKF1D(initial_state=initial_position[0], system_model=1, system_model_error=model_error[0],
                                 measurement_error=measurement_error[0],
                                 state_error_variance=1, residual_threshold=residual_threshold[0],
                                 adaptive_threshold=adaptive_threshold[0],
                                 estimator_parameter_count=estimator_parameter_count,
                                 forgetting_factor=forgetting_factor,
                                 region_factors=region_factors)

        self.rakf_cvm_y = RAKF1D(initial_state=initial_position[1], system_model=1, system_model_error=model_error[1],
                                 measurement_error=measurement_error[1],
                                 state_error_variance=1, residual_threshold=residual_threshold[1],
                                 adaptive_threshold=adaptive_threshold[1],
                                 estimator_parameter_count=estimator_parameter_count,
                                 forgetting_factor=forgetting_factor,
                                 region_factors=region_factors)

        self.rakf_cvm_z = RAKF1D(initial_state=initial_position[2], system_model=1, system_model_error=model_error[2],
                                 measurement_error=measurement_error[2],
                                 state_error_variance=100, residual_threshold=residual_threshold[2],
                                 adaptive_threshold=adaptive_threshold[2],
                                 estimator_parameter_count=estimator_parameter_count,
                                 forgetting_factor=forgetting_factor,
                                 region_factors=region_factors)

        # RAKF_CAM Creating an object for all axis (X, Y, Z)
        self.rakf_cam_x = RAKF1D(initial_state=initial_position[0], system_model=1, system_model_error=model_error[0],
                                 measurement_error=measurement_error[0],
                                 state_error_variance=1, residual_threshold=residual_threshold[0],
                                 adaptive_threshold=adaptive_threshold[0],
                                 estimator_parameter_count=estimator_parameter_count,
                                 forgetting_factor=forgetting_factor,
                                 region_factors=region_factors)

        self.rakf_cam_y = RAKF1D(initial_state=initial_position[1], system_model=1, system_model_error=model_error[1],
                                 measurement_error=measurement_error[1],
                                 state_error_variance=1, residual_threshold=residual_threshold[1],
                                 adaptive_threshold=adaptive_threshold[1],
                                 estimator_parameter_count=estimator_parameter_count,
                                 forgetting_factor=forgetting_factor,
                                 region_factors=region_factors)

        self.rakf_cam_z = RAKF1D(initial_state=initial_position[2], system_model=1, system_model_error=model_error[2],
                                 measurement_error=measurement_error[2],
                                 state_error_variance=100, residual_threshold=residual_threshold[2],
                                 adaptive_threshold=adaptive_threshold[2],
                                 estimator_parameter_count=estimator_parameter_count,
                                 forgetting_factor=forgetting_factor,
                                 region_factors=region_factors)

    def append_measurement(self, data):
        self.receive_data_queue.append(data=data)

    def run_on_measurements(self, measurement_x, measurement_y, measurement_z,
                            velocity_x=0, velocity_y=0, velocity_z=0,
                            acceleration_x=0, acceleration_y=0, acceleration_z=0,
                            timestamp_ms=0):

        # run RAKF CLM algorithm over measurements
        res_eqn_x, res_var_x = self.rakf_x.run(measurement_now=measurement_x, timestamp_ms=timestamp_ms,
                                               velocity=0, acceleration=0)
        res_eqn_y, res_var_y = self.rakf_y.run(measurement_now=measurement_y, timestamp_ms=timestamp_ms,
                                               velocity=0, acceleration=0)
        res_eqn_z, res_var_z = self.rakf_z.run(measurement_now=measurement_z, timestamp_ms=timestamp_ms,
                                               velocity=0, acceleration=0)

        # run RAKF_CVM algorithm over measurements
        res_eqn_cvm_x, res_var_cvm_x = self.rakf_cvm_x.run(measurement_now=measurement_x, timestamp_ms=timestamp_ms,
                                                           velocity=velocity_x, acceleration=0)
        res_eqn_cvm_y, res_var_cvm_y = self.rakf_cvm_y.run(measurement_now=measurement_y, timestamp_ms=timestamp_ms,
                                                           velocity=velocity_y, acceleration=0)
        res_eqn_cvm_z, res_var_cvm_z = self.rakf_cvm_z.run(measurement_now=measurement_z, timestamp_ms=timestamp_ms,
                                                           velocity=velocity_z, acceleration=0)
        # run RAKF CAM algorithm over measurements
        res_eqn_cam_x, res_var_cam_x = self.rakf_cam_x.run(measurement_now=measurement_x, timestamp_ms=timestamp_ms,
                                                           velocity=velocity_x, acceleration=acceleration_x)
        res_eqn_cam_y, res_var_cam_y = self.rakf_cam_y.run(measurement_now=measurement_y, timestamp_ms=timestamp_ms,
                                                           velocity=velocity_y, acceleration=acceleration_y)
        res_eqn_cam_z, res_var_cam_z = self.rakf_cam_z.run(measurement_now=measurement_z, timestamp_ms=timestamp_ms,
                                                           velocity=velocity_z, acceleration=acceleration_z)

        # Get timedelta based on timestamp and update velocity
        if self.time_previous < 0:
            timedelta = 0.0
            self.vel_x = 0.0
            self.vel_y = 0.0
            self.vel_z = 0.0
        else:
            timedelta = timestamp_ms - self.time_previous
            self.vel_x = ((self.pos_x - res_var_x["state_model"]) / timedelta)
            self.vel_y = ((self.pos_y - res_var_y["state_model"]) / timedelta)
            self.vel_z = ((self.pos_z - res_var_z["state_model"]) / timedelta)

        self.time_previous = timestamp_ms

        # update current position
        self.pos_x = res_var_x["state_model"]
        self.pos_y = res_var_y["state_model"]
        self.pos_z = res_var_z["state_model"]

        return [res_var_x, res_var_y, res_var_z, res_var_cvm_x, res_var_cvm_y, res_var_cvm_z, res_var_cam_x,
                res_var_cam_y, res_var_cam_z]

    def run(self):

        # dequeue receive data from the queue, extract timestamp and measurement information
        if not self.receive_data_queue.is_empty():
            data_item = self.receive_data_queue.pop()
            timestamp_ms = data_item["timestamp"]
            measurements = data_item["coordinates"]
        else:
            timestamp_ms = int(round(time.time() * 1000))
            measurements = {"x": self.pos_x, "y": self.pos_y, "z": self.pos_z}

        # run RAKF algorithm over measurements
        res_eqn_x, res_var_x = self.rakf_x.run(measurement_now=measurements["x"], timestamp_ms=timestamp_ms)
        res_eqn_y, res_var_y = self.rakf_y.run(measurement_now=measurements["y"], timestamp_ms=timestamp_ms)
        res_eqn_z, res_var_z = self.rakf_z.run(measurement_now=measurements["z"], timestamp_ms=timestamp_ms)

        # update current position
        self.pos_x = res_var_x["state_model"]
        self.pos_y = res_var_y["state_model"]
        self.pos_z = res_var_z["state_model"]

        # send data to influxdb using telegraph via mqtt
        if telegraph_mqtt_client is not None:
            send_string = f'location,device_id=1 x_f={res_var_x["state_model"]},y_f={res_var_y["state_model"]},' \
                          f'z_f={res_var_z["state_model"]},x={measurements["x"]},y={measurements["y"]},z={measurements["z"]}'
            telegraph_mqtt_client.publish(topic="uptime/telemetry", payload=send_string)

        # filtered_string = f'location,device_id=1 x_f={res_var_x["state_model"]},y_f={res_var_y["state_model"]},' \
        #                   f'z_f={res_var_z["state_model"]}'

        # await asyncio.sleep(self.loop_time_sec)
