import os
import yaml
import numpy

from filterpy.kalman import KalmanFilter


class KF3D:
    def __init__(self, json_file_name, tag_id):
        self.kf = KalmanFilter (dim_x=3, dim_z=3)

        model_error = []
        measurement_error = []
        initial_position = []

        if os.path.exists( json_file_name ):
            with open( json_file_name, 'r' ) as json_file:
                # client_json = json.load( json_file )
                client_json = yaml.load( json_file, Loader=yaml.FullLoader )
                for entry in client_json["rakf_config"]:
                    if entry["tag_id"] == tag_id:
                        # --- add more attributes ------
                        for item in entry["model_error"]:
                            model_error.append( item )
                        for item in entry["measurement_error"]:
                            measurement_error.append( item )
                        for item in entry["initial_position"]:
                            initial_position.append(item)
        else:
            print("No tag with id" + tag_id + "found")
            exit(-1)

        # initializing KalmanFilter
        self.kf.x = numpy.array([[initial_position[0]],         # position_x
                                   [initial_position[1]],       # position_y
                                   [initial_position[2]]])      # position_z

        self.kf.F = numpy.array([[1., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 1.]])

        self.kf.H = numpy.array([[1., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 1.]])

        self.kf.P = numpy.array([[1., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 1.]])

        self.kf.R = numpy.array([[measurement_error[0], 0., 0.],
                                 [0., measurement_error[1], 0.],
                                 [0., 0., measurement_error[2]]])

        self.kf.Q = numpy.array([[model_error[0], 0., 0.],
                                 [0., model_error[0], 0.],          # model_error[0]
                                 [0., 0., model_error[0]]])

    def run_on_measurements(self, measurement_x, measurement_y, measurement_z, timestamp_ms=0):
        z = numpy.array([[measurement_x],  # position_x
                         [measurement_y],  # position_y
                         [measurement_z]])  # position_z
        # Prediction step
        self.kf.predict()
        # Update step
        self.kf.update(z)

        return [self.kf.x[0],self.kf.x[1], self.kf.x[2]]




