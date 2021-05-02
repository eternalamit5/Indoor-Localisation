# Python code for 2D random walk.
import random
import time
import matplotlib.pyplot as plt
import numpy
import math
from Algorithm.OutlierGenerator import OutlierGenerator


class Sigmoid:
    def __init__(self, mid_point=0, steepness=0.5, max_value=1, level_shift=0):
        self.mid_point = mid_point
        self.steepness = steepness
        self.max_value = max_value
        self.level_shift = level_shift

    def generate(self, x):
        denom = 1 + math.exp(-1.0 * self.steepness * (x - self.mid_point))
        return (self.max_value / denom) + self.level_shift


class WalkAngleGenerator:
    def __init__(self, mid_point=0, steepness=0.5, max_value=1, level_shift=0):
        self.sigmoid = Sigmoid(mid_point=mid_point, steepness=steepness, max_value=max_value, level_shift=level_shift)

    def get_angle_deviation(self, velocity):
        return self.sigmoid.generate(-1.0 * velocity)

    def generate(self, angle, velocity):
        one_standard_deviation = 0.341  #
        return numpy.random.normal(loc=angle,
                                   scale=self.get_angle_deviation(velocity=velocity) * one_standard_deviation)


class WalkPatternGenerator:

    # def factory(self,file_name):
    #     walkers = []
    #     if os.path.exists( file_name ):
    #         with open( file_name, 'r' ) as json_file:
    #             client_json = yaml.load( json_file, Loader=yaml.FullLoader )
    #             for entry in client_json["walking-persons"]:
    #                 outlier = []
    #                 walker = None
    #                 id = entry["id"]
    #                 if entry["outliers"]:
    #                     for outlier in entry["outliers"]:
    #                         outlier.append(OutlierGenerator( mean=outlier["mean"], standard_deviation=outlier["standard-deviation"],
    #                                                          number_of_outliers=outlier["number-of-outlier"], sample_size=outlier["sample-size"] ))
    #
    #                 if entry["walker-attributes"]:
    #                     walker_attrib = entry["walker-attributes"]
    #                     sigmoid_attrib =  walker_attrib["sigmoid-attributes"]
    #                     walker = WalkPatternGenerator( boundary=walker_attrib["walk-boundary"], avg_speed_mps=walker_attrib["avg-walk-speed-mps"],
    #                                                          walk_dimension=walker_attrib["walk-dimension"], outlier_model_x=outlier[0],
    #                                                          outlier_model_y=outlier[1], outlier_model_z=outlier[2], mid_point=sigmoid_attrib["mid-point"],
    #                                                          steepness=sigmoid_attrib["steepness"] )
    #                 walkers.append({"id":id, "model":walker})

    def __init__(self, boundary=100, avg_speed_mps=1, walk_dimension=3, outlier_model_x=None, outlier_model_y=None,
                 outlier_model_z=None, mid_point=0, steepness=0.5, surface=None):
        """
        :rtype: NA
        :param boundary: boundary within which walk is to be performed (in meters), default value =100
        :param avg_speed_mps: max speed of walk in 'meter per second', default value =1
        :param walk_dimension: type of walk, default value ='all'
                        # 'all' Toss -based walk in all direction (forward, backward, sidewards, stop)
                        # 'forward': Toss-based walk in forward direction only ( forward or stop)
        :param outlier_model_x: Outlier model for x axis, default value =None
        :param outlier_model_y: Outlier model for y axis, default value =None
        :param outlier_model_z: Outlier model for z axis, default value =None
        """
        self.x_pos = 0
        self.y_pos = 0
        self.z_pos = 0

        self.x_outlier_pos = 0
        self.y_outlier_pos = 0
        self.z_outlier_pos = 0

        self.x_pos_prev = 0
        self.y_pos_prev = 0
        self.z_pos_prev = 0

        self.x_velocity = 0
        self.y_velocity = 0
        self.z_velocity = 0

        self.x_velocity_prev = 0
        self.y_velocity_prev = 0
        self.z_velocity_prev = 0

        self.x_acceleration = 0
        self.y_acceleration = 0
        self.z_acceleration = 0

        self.x_acceleration_prev = 0
        self.y_acceleration_prev = 0
        self.z_acceleration_prev = 0

        self.x_jerk = 0
        self.y_jerk = 0
        self.z_jerk = 0

        self.step_size = 0
        self.x_step_length = 0
        self.y_step_length = 0
        self.z_step_length = 0

        self.time_now = 0
        self.time_past = 0

        self.boundary = boundary
        self.avg_speed_mps = avg_speed_mps
        self.walk_dimension = walk_dimension

        self.walk_angle = 0
        self.walk_angle_gen = WalkAngleGenerator(mid_point=mid_point, steepness=steepness,
                                                 max_value=math.radians(135.0), level_shift=math.radians(45.0))

        self.outlier_model_x = outlier_model_x
        self.outlier_model_y = outlier_model_y
        self.outlier_model_z = outlier_model_z

        self.surface = surface

    def update(self, tdelta=-1):
        if self.walk_dimension == 1:
            self.update3d(tdelta)
            return {"x_pos": self.x_pos, "x_outlier_pos": self.x_outlier_pos, "x_velocity": self.x_velocity,
                    "x_acceleration": self.x_acceleration,
                    "x_jerk": self.x_jerk}
        elif self.walk_dimension == 2:
            self.update3d(tdelta)
            return {"x_pos": self.x_pos, "y_pos": self.y_pos, "x_outlier_pos": self.x_outlier_pos,
                    "y_outlier_pos": self.y_outlier_pos,
                    "x_velocity": self.x_velocity, "y_velocity": self.y_velocity, "x_acceleration": self.x_acceleration,
                    "y_acceleration": self.y_acceleration,
                    "x_jerk": self.x_jerk, "y_jerk": self.y_jerk}
        else:
            return self.update3d(tdelta)

    def update3d(self, tdelta=-1):
        # calculate loop time
        if tdelta > 0:
            timedelta = tdelta
        elif self.time_now == 0 and self.time_past == 0:
            self.time_now = time.time()
            timedelta = 0.01
        else:
            self.time_now = time.time()
            timedelta = self.time_now - self.time_past

        distance_in_sample_time = self.avg_speed_mps * timedelta

        # calculate instantaneous velocity, based on step size calculated in previous iteration and take direction decision
        self.walk_angle = self.walk_angle_gen.generate(self.walk_angle, self.step_size / timedelta)

        # step size decision
        self.step_size = random.uniform(self.step_size, distance_in_sample_time * 0.682)

        # step length in each of the axis
        self.x_step_length = self.step_size * math.cos(self.walk_angle)
        self.y_step_length = self.step_size * math.sin(self.walk_angle)
        self.z_step_length = 0  # math.sqrt(math.pow(x_step_length,2) + math.pow(x_step_length,2) -math.pow(self.step_size,2))
        # self.z_step_length = math.sin(math.sqrt((math.pow(self.x_step_length,2) + math.pow(self.y_step_length,2)))) # todo write logic for z_step_length based on angle

        # walk based on step size calculated in each direction
        self.x_pos = self.x_pos_prev + self.x_step_length
        self.y_pos = self.y_pos_prev + self.y_step_length
        self.z_pos = self.z_pos_prev + self.z_step_length

        # check for outlier model
        if self.outlier_model_x is not None:
            self.x_outlier_pos = self.x_pos + self.outlier_model_x.generate()

        if self.outlier_model_y is not None:
            self.y_outlier_pos = self.y_pos + self.outlier_model_y.generate()

        if self.outlier_model_z is not None:
            self.z_outlier_pos = self.z_pos + self.outlier_model_z.generate()

        # calculate acceleration
        # self.x_acceleration = (self.x_velocity - self.x_velocity_prev) / timedelta
        # self.y_acceleration = (self.y_velocity - self.y_velocity_prev) / timedelta
        # self.z_acceleration = (self.z_velocity - self.z_velocity_prev) / timedelta
        self.x_acceleration = ((self.x_pos - self.x_pos_prev) / (timedelta ** 2)) + numpy.random.normal(loc=0,
                                                                                                        scale=0.5)
        self.y_acceleration = ((self.y_pos - self.y_pos_prev) / (timedelta ** 2)) + numpy.random.normal(loc=0,
                                                                                                        scale=0.5)
        self.z_acceleration = ((self.z_pos - self.z_pos_prev) / (timedelta ** 2)) + numpy.random.normal(loc=0,
                                                                                                        scale=0.5)
        # calculate velocity
        # self.x_velocity = (self.x_pos - self.x_pos_prev) / timedelta + numpy.random.normal(0, 0.1, 1)
        # self.y_velocity = (self.y_pos - self.y_pos_prev) / timedelta
        # self.z_velocity = (self.z_pos - self.z_pos_prev) / timedelta
        self.x_velocity = self.x_acceleration * timedelta
        self.y_velocity = self.y_acceleration * timedelta
        self.z_velocity = self.z_acceleration * timedelta

        # calculate jerk
        self.x_jerk = (self.x_acceleration - self.x_acceleration_prev) / timedelta
        self.y_jerk = (self.y_acceleration - self.y_acceleration_prev) / timedelta
        self.z_jerk = (self.z_acceleration - self.z_acceleration_prev) / timedelta

        # prepare for next iteration
        self.x_pos_prev = self.x_pos
        self.y_pos_prev = self.y_pos
        self.z_pos_prev = self.z_pos

        self.x_velocity_prev = self.x_velocity
        self.y_velocity_prev = self.y_velocity
        self.z_velocity_prev = self.z_velocity

        self.x_acceleration_prev = self.x_acceleration
        self.y_acceleration_prev = self.y_acceleration
        self.z_acceleration_prev = self.z_acceleration

        self.time_past = self.time_now

        return {"x_pos": self.x_pos, "y_pos": self.y_pos, "z_pos": self.z_pos, "x_outlier_pos": self.x_outlier_pos,
                "y_outlier_pos": self.y_outlier_pos,
                "z_outlier_pos": self.z_outlier_pos, "x_velocity": self.x_velocity, "y_velocity": self.y_velocity,
                "z_velocity": self.z_velocity, "x_acceleration": self.x_acceleration,
                "y_acceleration": self.y_acceleration,
                "z_acceleration": self.z_acceleration, "x_jerk": self.x_jerk, "y_jerk": self.y_jerk,
                "z_jerk": self.z_jerk}

    def get_states(self):
        return {"x_pos": self.x_pos, "y_pos": self.y_pos, "z_pos": self.z_pos, "x_outlier_pos": self.x_outlier_pos,
                "y_outlier_pos": self.y_outlier_pos,
                "z_outlier_pos": self.z_outlier_pos, "x_velocity": self.x_velocity, "y_velocity": self.y_velocity,
                "z_velocity": self.z_velocity, "x_acceleration": self.x_acceleration,
                "y_acceleration": self.y_acceleration,
                "z_acceleration": self.z_acceleration, "x_jerk": self.x_jerk, "y_jerk": self.y_jerk,
                "z_jerk": self.z_jerk}


# ----------------------------------------- Test -------------------------------------------------
def plot2d(x, y, title="", legend="", overwrite=True):
    if not overwrite:
        fig = plt.figure()
        subplot1 = fig.add_subplot(111)
        subplot1.title(title)
        subplot1.plot(x, y, label=legend)
    else:
        plt.title(title)
        plt.plot(x, y, label=legend)


def plot3d(x, y, z, title="", legend="", overwrite=False):
    if not overwrite:
        fig = plt.figure()
        subplot1 = fig.add_subplot(111, projection='3d')
        subplot1.title(title)
        subplot1.plot(x, y, z, label=legend)
    else:
        plt.title(title)
        plt.plot(x, y, label=legend)


def walk_pattern_test():
    number_of_samples = 100
    num_of_outlier = 0
    outlier_x_gen = OutlierGenerator(mean=0, standard_deviation=5, number_of_outliers=num_of_outlier,
                                     sample_size=number_of_samples)
    outlier_y_gen = OutlierGenerator(mean=0, standard_deviation=5, number_of_outliers=num_of_outlier,
                                     sample_size=number_of_samples)
    outlier_z_gen = OutlierGenerator(mean=0, standard_deviation=5, number_of_outliers=num_of_outlier,
                                     sample_size=number_of_samples)
    walker = WalkPatternGenerator(boundary=100, avg_speed_mps=1.7, walk_dimension=3, outlier_model_x=outlier_x_gen,
                                  outlier_model_y=outlier_y_gen, outlier_model_z=outlier_z_gen,
                                  mid_point=2, steepness=0.2)

    postion_raw_x = numpy.zeros(number_of_samples)
    position_raw_y = numpy.zeros(number_of_samples)
    position_raw_z = numpy.zeros(number_of_samples)
    position_raw_with_outlier_x = numpy.zeros(number_of_samples)
    position_raw_with_outlier_y = numpy.zeros(number_of_samples)
    position_raw_with_outlier_z = numpy.zeros(number_of_samples)
    # velocity_raw_with_outlier_x = numpy.zeros( number_of_samples )
    # velocity_raw_with_outlier_y = numpy.zeros( number_of_samples )
    # velocity_raw_with_outlier_z = numpy.zeros( number_of_samples )
    # acceleration_raw_with_outlier_x = numpy.zeros( number_of_samples )
    # acceleration_raw_with_outlier_y = numpy.zeros( number_of_samples )
    # acceleration_raw_with_outlier_z = numpy.zeros( number_of_samples )
    # jerk_raw_with_outlier_x = numpy.zeros( number_of_samples )
    # jerk_raw_with_outlier_y = numpy.zeros( number_of_samples )
    # jerk_raw_with_outlier_z = numpy.zeros( number_of_samples )
    input_sample = numpy.zeros(number_of_samples)
    for i in range(1, number_of_samples):
        states = walker.update(0.7)

        # update states
        postion_raw_x[i] = states["x_pos"]
        position_raw_y[i] = states["y_pos"]
        position_raw_z[i] = states["z_pos"]
        position_raw_with_outlier_x[i] = states["x_outlier_pos"]
        position_raw_with_outlier_y[i] = states["y_outlier_pos"]
        position_raw_with_outlier_z[i] = states["z_outlier_pos"]
        # velocity_raw_with_outlier_x[i] = states["x_velocity"]
        # velocity_raw_with_outlier_y[i] = states["y_velocity"]
        # velocity_raw_with_outlier_z[i] = states["z_velocity"]
        # acceleration_raw_with_outlier_x[i] = states["x_acceleration"]
        # acceleration_raw_with_outlier_y[i] = states["y_acceleration"]
        # acceleration_raw_with_outlier_z[i] = states["z_acceleration"]
        # jerk_raw_with_outlier_x[i] = states["x_jerk"]
        # jerk_raw_with_outlier_y[i] = states["y_jerk"]
        # jerk_raw_with_outlier_z[i] = states["z_jerk"]
        input_sample[i] = i

    fig1 = plt.figure()
    fig1plot1 = fig1.add_subplot(311)
    fig1plot1.title.set_text("Random Walk x-axis")
    fig1plot1.set_xlabel("steps")
    fig1plot1.set_ylabel("position")
    fig1plot1.plot(input_sample, position_raw_with_outlier_x, label="outlier", color="r", linestyle="-", marker=".")
    fig1plot1.plot(input_sample, postion_raw_x, label="actual", color="g", linestyle="--", marker=".")

    fig1plot2 = fig1.add_subplot(312)
    fig1plot2.title.set_text("Random Walk y-axis")
    fig1plot2.set_xlabel("steps")
    fig1plot2.set_ylabel("position")
    fig1plot2.plot(input_sample, position_raw_with_outlier_y, label="outlier", color="r", linestyle="-", marker=".")
    fig1plot2.plot(input_sample, position_raw_y, label="actual", color="g", linestyle="--", marker=".")

    fig1plot3 = fig1.add_subplot(313)
    fig1plot3.title.set_text("Random Walk z-axis")
    fig1plot3.set_xlabel("steps")
    fig1plot3.set_ylabel("position")
    fig1plot3.plot(input_sample, position_raw_with_outlier_z, label="outlier", color="r", linestyle="-", marker=".")
    fig1plot3.plot(input_sample, position_raw_z, label="actual", color="g", linestyle="--", marker=".")

    fig2 = plt.figure()
    fig2plot1 = fig2.add_subplot(111, projection='3d')
    fig2plot1.title.set_text("Random Walk 3D")
    fig2plot1.set_xlabel("x position")
    fig2plot1.set_ylabel("y position")
    fig2plot1.set_zlabel("z position")
    fig2plot1.plot(position_raw_with_outlier_x, position_raw_with_outlier_y, position_raw_with_outlier_z,
                   label="outlier", color="r", linestyle="--")
    fig2plot1.plot(postion_raw_x, position_raw_y, position_raw_z, label="actual", color="g", linestyle="--")

    fig3 = plt.figure()
    fig3plot1 = fig3.add_subplot(111)
    fig3plot1.title.set_text("Random Walk 2D")
    fig3plot1.set_xlabel("x position")
    fig3plot1.set_ylabel("y position")
    fig3plot1.plot(position_raw_with_outlier_x, position_raw_with_outlier_y, label="outlier", color="r", linestyle="--")
    fig3plot1.plot(postion_raw_x, position_raw_y, label="actual", color="g", linestyle="--")

    plt.legend()
    plt.show()


def sigmoid_wave_test():
    number_of_samples = 100

    sigmoid_1 = Sigmoid(steepness=0.1)
    sigmoid_1_out = numpy.zeros(number_of_samples)
    sigmoid_1_in = numpy.zeros(number_of_samples)

    start_val = int(number_of_samples / 2)
    stop_val = int((-1 * start_val) - 1)
    for i in range(start_val, stop_val, -1):
        sigmoid_1_out[i - start_val] = sigmoid_1.generate(i)
        sigmoid_1_in[i - start_val] = i

    plt.title("Sigmoid-1 ($n = " + str(number_of_samples) + "$ steps)")
    plt.xlabel("Input samples")
    plt.ylabel("Output values")
    plt.plot(sigmoid_1_in, sigmoid_1_out)
    plt.show()


def walk_angle_test():
    mid_point = -2
    steepness = 0.5
    angle_deviation_degrees = 10.0
    max_speed_mps = 40.0

    walk_angle_gen = WalkAngleGenerator(mid_point=mid_point, steepness=steepness, max_value=math.radians(135),
                                        level_shift=math.radians(45))

    walk_angle_result = numpy.zeros(40)
    velocity_in = numpy.zeros(40)

    prev_angle = 0.0
    for i in range(0, 40):
        walk_angle_result[i] = math.degrees(walk_angle_gen.generate(prev_angle, i * 1.0))
        prev_angle = 0  # math.radians(walk_angle_result[i])
        velocity_in[i] = i

    plt.title("Walk angle")
    plt.xlabel("Velocity")
    plt.ylabel("Angle in degrees")
    plt.plot(velocity_in, walk_angle_result)
    plt.show()


if __name__ == '__main__':
    walk_pattern_test()
