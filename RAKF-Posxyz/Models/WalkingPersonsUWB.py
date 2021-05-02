import matplotlib.pyplot as plt
import numpy
from Algorithm.RAKF3D import RAKF3D
from Algorithm.WalkingPatternGenerator import WalkPatternGenerator
from Algorithm.OutlierGenerator import OutlierGenerator
from Algorithm.KF3D import KF3D

from sklearn.metrics import r2_score, max_error, mean_squared_error
from tabulate import tabulate
import time

if __name__ == '__main__':
    sample_time = 0.1
    sample_timestamp = sample_time

    number_of_samples = 100
    num_of_outlier = 75
    outlier_sample_size = 100
    outlier_x_gen = OutlierGenerator(mean=0, standard_deviation=2, number_of_outliers=num_of_outlier,
                                     sample_size=outlier_sample_size)
    outlier_y_gen = OutlierGenerator(mean=0, standard_deviation=2, number_of_outliers=num_of_outlier,
                                     sample_size=outlier_sample_size)
    outlier_z_gen = OutlierGenerator(mean=0, standard_deviation=1, number_of_outliers=num_of_outlier,
                                     sample_size=outlier_sample_size)
    walker = WalkPatternGenerator(boundary=100, avg_speed_mps=3, walk_dimension=3, outlier_model_x=outlier_x_gen,
                                  outlier_model_y=outlier_y_gen, outlier_model_z=outlier_z_gen,
                                  mid_point=2, steepness=0.4)
    # Kalman object creation
    kf_3d = KF3D(json_file_name="../Config/RAKF3D.yaml", tag_id=1)

    # RAKF object creation
    rakf_3d = RAKF3D(json_file_name="../Config/RAKF3D.yaml", tag_id=1)

    # Kalman initialization
    kf_position_x = numpy.zeros(number_of_samples)
    kf_position_y = numpy.zeros(number_of_samples)
    kf_position_z = numpy.zeros(number_of_samples)

    # RAKF initialization
    rakf_position_x = numpy.zeros(number_of_samples)
    rakf_position_y = numpy.zeros(number_of_samples)
    rakf_position_z = numpy.zeros(number_of_samples)
    rakf_state_estimation_x = numpy.zeros(number_of_samples)
    rakf_state_estimation_y = numpy.zeros(number_of_samples)
    rakf_state_estimation_z = numpy.zeros(number_of_samples)
    rakf_gain_x = numpy.zeros(number_of_samples)
    rakf_gain_y = numpy.zeros(number_of_samples)
    rakf_gain_z = numpy.zeros(number_of_samples)

    # RAKF CVM initialization
    rakf_cvm_position_x = numpy.zeros(number_of_samples)
    rakf_cvm_position_y = numpy.zeros(number_of_samples)
    rakf_cvm_position_z = numpy.zeros(number_of_samples)
    rakf_cvm_state_estimation_x = numpy.zeros(number_of_samples)
    rakf_cvm_state_estimation_y = numpy.zeros(number_of_samples)
    rakf_cvm_state_estimation_z = numpy.zeros(number_of_samples)
    rakf_cvm_gain_x = numpy.zeros(number_of_samples)
    rakf_cvm_gain_y = numpy.zeros(number_of_samples)
    rakf_cvm_gain_z = numpy.zeros(number_of_samples)

    # RAKF CAM initialization
    rakf_cam_position_x = numpy.zeros(number_of_samples)
    rakf_cam_position_y = numpy.zeros(number_of_samples)
    rakf_cam_position_z = numpy.zeros(number_of_samples)
    rakf_cam_state_estimation_x = numpy.zeros(number_of_samples)
    rakf_cam_state_estimation_y = numpy.zeros(number_of_samples)
    rakf_cam_state_estimation_z = numpy.zeros(number_of_samples)
    rakf_cam_gain_x = numpy.zeros(number_of_samples)
    rakf_cam_gain_y = numpy.zeros(number_of_samples)
    rakf_cam_gain_z = numpy.zeros(number_of_samples)

    # RAW data initialization
    position_raw_x = numpy.zeros(number_of_samples)
    position_raw_y = numpy.zeros(number_of_samples)
    position_raw_z = numpy.zeros(number_of_samples)
    position_raw_with_outlier_x = numpy.zeros(number_of_samples)
    position_raw_with_outlier_y = numpy.zeros(number_of_samples)
    position_raw_with_outlier_z = numpy.zeros(number_of_samples)
    velocity_raw_with_outlier_x = numpy.zeros(number_of_samples)
    velocity_raw_with_outlier_y = numpy.zeros(number_of_samples)
    velocity_raw_with_outlier_z = numpy.zeros(number_of_samples)
    acceleration_raw_with_outlier_x = numpy.zeros(number_of_samples)
    acceleration_raw_with_outlier_y = numpy.zeros(number_of_samples)
    acceleration_raw_with_outlier_z = numpy.zeros(number_of_samples)
    jerk_raw_with_outlier_x = numpy.zeros(number_of_samples)
    jerk_raw_with_outlier_y = numpy.zeros(number_of_samples)
    jerk_raw_with_outlier_z = numpy.zeros(number_of_samples)
    input_sample = numpy.zeros(number_of_samples)

    # Variable to calculate execution time of the algorithm
    kf_exec_time = numpy.zeros(number_of_samples)
    rakf_exec_time = numpy.zeros(number_of_samples)

    for i in range(1, number_of_samples):
        states = walker.update(sample_time)
        # KALMAN run_on_measurements
        start_time = time.time()
        result_kf = kf_3d.run_on_measurements(measurement_x=states["x_outlier_pos"],
                                              measurement_y=states["y_outlier_pos"],
                                              measurement_z=states["z_outlier_pos"],
                                              timestamp_ms=sample_timestamp)
        kf_exec_time[i] = time.time() - start_time

        # RAKF run_on_measurements
        start_time = time.time()
        result_rakf = rakf_3d.run_on_measurements(measurement_x=states["x_outlier_pos"],
                                                  measurement_y=states["y_outlier_pos"],
                                                  measurement_z=states["z_outlier_pos"],
                                                  velocity_x=states["x_velocity"],
                                                  velocity_y=states["y_velocity"],
                                                  velocity_z=states["z_velocity"],
                                                  acceleration_x=states["x_acceleration"],
                                                  acceleration_y=states["y_acceleration"],
                                                  acceleration_z=states["z_acceleration"],
                                                  timestamp_ms=sample_timestamp)
        rakf_exec_time[i] = time.time() - start_time

        sample_timestamp = sample_timestamp + sample_time

        # KF update states
        kf_position_x[i] = result_kf[0]
        kf_position_y[i] = result_kf[1]
        kf_position_z[i] = result_kf[2]

        # RAKF update states
        rakf_position_x[i] = result_rakf[0]["state_model"]
        rakf_position_y[i] = result_rakf[1]["state_model"]
        rakf_position_z[i] = result_rakf[2]["state_model"]
        rakf_state_estimation_x[i] = result_rakf[0]["state_estimation"]
        rakf_state_estimation_y[i] = result_rakf[1]["state_estimation"]
        rakf_state_estimation_z[i] = result_rakf[2]["state_estimation"]
        rakf_gain_x[i] = result_rakf[0]["delta_state_estimate"]
        rakf_gain_y[i] = result_rakf[1]["delta_state_estimate"]
        rakf_gain_z[i] = result_rakf[2]["delta_state_estimate"]

        # RAKF CVM update states
        rakf_cvm_position_x[i] = result_rakf[3]["state_model"]
        rakf_cvm_position_y[i] = result_rakf[4]["state_model"]
        rakf_cvm_position_z[i] = result_rakf[5]["state_model"]
        rakf_cvm_state_estimation_x[i] = result_rakf[3]["state_estimation"]
        rakf_cvm_state_estimation_y[i] = result_rakf[4]["state_estimation"]
        rakf_cvm_state_estimation_z[i] = result_rakf[5]["state_estimation"]
        rakf_cvm_gain_x[i] = result_rakf[3]["delta_state_estimate"]
        rakf_cvm_gain_y[i] = result_rakf[4]["delta_state_estimate"]
        rakf_cvm_gain_z[i] = result_rakf[5]["delta_state_estimate"]

        # RAKF CAM update states
        rakf_cam_position_x[i] = result_rakf[6]["state_model"]
        rakf_cam_position_y[i] = result_rakf[7]["state_model"]
        rakf_cam_position_z[i] = result_rakf[8]["state_model"]
        rakf_cam_state_estimation_x[i] = result_rakf[6]["state_estimation"]
        rakf_cam_state_estimation_y[i] = result_rakf[7]["state_estimation"]
        rakf_cam_state_estimation_z[i] = result_rakf[8]["state_estimation"]
        rakf_cam_gain_x[i] = result_rakf[6]["delta_state_estimate"]
        rakf_cam_gain_y[i] = result_rakf[7]["delta_state_estimate"]
        rakf_cam_gain_z[i] = result_rakf[8]["delta_state_estimate"]

        position_raw_x[i] = states["x_pos"]
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

    # Error array (TruePosition - RAKF algorithm) --------------------------------------------------------------
    # X-AXIS
    position_raw_with_outlier_error_x = numpy.subtract(position_raw_x,
                                                       position_raw_with_outlier_x)  # Raw position with outlier error
    kf_error_x = numpy.subtract(position_raw_x, kf_position_x)                                  # KF error
    rakf_state_estimation__error_x = numpy.subtract(position_raw_x, rakf_state_estimation_x)    # WLS error
    rakf_error_x = numpy.subtract(position_raw_x, rakf_position_x)                              # RAKF  error
    rakf_cvm_error_x = numpy.subtract(position_raw_x, rakf_cvm_position_x)                      # RAKF CVM error
    rakf_cam_error_x = numpy.subtract(position_raw_x, rakf_cam_position_x)                      # RAKF CAM error

    # Y-AXIS
    position_raw_with_outlier_error_y = numpy.subtract(position_raw_y,
                                                       position_raw_with_outlier_y)  # Raw position with outlier error
    kf_error_y = numpy.subtract(position_raw_y, kf_position_y)                                  # KF error
    rakf_state_estimation__error_y = numpy.subtract(position_raw_y, rakf_state_estimation_y)    # WLS error
    rakf_error_y = numpy.subtract(position_raw_y, rakf_position_y)                              # RAKF algorithm error
    rakf_cvm_error_y = numpy.subtract(position_raw_y, rakf_cvm_position_y)                      # RAKF CVM error
    rakf_cam_error_y = numpy.subtract(position_raw_y, rakf_cam_position_y)                      # RAKF CVM error

    # Z-AXIS
    position_raw_with_outlier_error_z = numpy.subtract(position_raw_z,
                                                       position_raw_with_outlier_z)  # Raw position with outlier error
    kf_error_z = numpy.subtract(position_raw_z, kf_position_z)                                  # KF error
    rakf_state_estimation__error_z = numpy.subtract(position_raw_z, rakf_state_estimation_z)    # WLS error
    rakf_error_z = numpy.subtract(position_raw_z, rakf_position_z)                              # RAKF algorithm error
    rakf_cvm_error_z = numpy.subtract(position_raw_z, rakf_cvm_position_z)                      # RAKF CVM error
    rakf_cam_error_z = numpy.subtract(position_raw_z, rakf_cam_position_z)  # RAKF CVM error

    # TABLE -------------------------------------------------------------------------------------
    # ERROR TABLE data
    table_format = "fancy_grid"
    table_header = ['ERROR DATA ANALYSIS',
                    'X-AXIS\nMEAN',
                    'X-AXIS\nSTD',
                    'Y-AXIS\nMEAN',
                    'Y-AXIS\nSTD',
                    'Z-AXIS\nMEAN',
                    'Z-AXIS\nSTD']
    table_content = [["RAW POSITION WITH OUTLIERS",
                      numpy.mean(position_raw_with_outlier_error_x), numpy.std(position_raw_with_outlier_error_x),
                      numpy.mean(position_raw_with_outlier_error_y), numpy.std(position_raw_with_outlier_error_y),
                      numpy.mean(position_raw_with_outlier_error_z), numpy.std(position_raw_with_outlier_error_z)],
                     # ["KF ALGORITHM",
                     #  numpy.mean(kf_error_x), numpy.std(kf_error_x),
                     #  numpy.mean(kf_error_y), numpy.std(kf_error_y),
                     #  numpy.mean(kf_error_z), numpy.std(kf_error_z)],
                     # ["STATE ESTIMATION ALGORITHM (WLS)",
                     #  numpy.mean(rakf_state_estimation__error_x), numpy.std(rakf_state_estimation__error_x),
                     #  numpy.mean(rakf_state_estimation__error_y), numpy.std(rakf_state_estimation__error_y),
                     #  numpy.mean(rakf_state_estimation__error_z), numpy.std(rakf_state_estimation__error_z)],
                     # ["RAKF CLM ALGORITHM",
                     #  numpy.mean(rakf_error_x), numpy.std(rakf_error_x),
                     #  numpy.mean(rakf_error_y), numpy.std(rakf_error_y),
                     #  numpy.mean(rakf_error_z), numpy.std(rakf_error_z)],
                     ["RAKF CVM ALGORITHM",
                      numpy.mean(rakf_cvm_error_x), numpy.std(rakf_cvm_error_x),
                      numpy.mean(rakf_cvm_error_y), numpy.std(rakf_cvm_error_y),
                      numpy.mean(rakf_cvm_error_z), numpy.std(rakf_cvm_error_z)],
                     ["RAKF CAM ALGORITHM",
                      numpy.mean(rakf_cam_error_x), numpy.std(rakf_cam_error_x),
                      numpy.mean(rakf_cam_error_y), numpy.std(rakf_cam_error_y),
                      numpy.mean(rakf_cam_error_z), numpy.std(rakf_cam_error_z)]
                     ]
    print(tabulate(table_content, table_header, table_format))

    # COEFFICIENT OF DETERMINATION TABLE
    table_format = "fancy_grid"
    table_header = ['COEFFICIENT OF DETERMINATION R^2',
                    'X-AXIS',
                    'Y-AXIS',
                    'Z-AXIS']
    table_content = [["RAW POSITION WITH OUTLIERS",
                      r2_score(position_raw_x, position_raw_with_outlier_x),
                      r2_score(position_raw_y, position_raw_with_outlier_y),
                      r2_score(position_raw_z, position_raw_with_outlier_z)],
                     # ["KF ALGORITHM",
                     #  r2_score(position_raw_x, kf_position_x),
                     #  r2_score(position_raw_y, kf_position_y),
                     #  r2_score(position_raw_z, kf_position_z)],
                     # ["STATE ESTIMATION ALGORITHM (WLS)",
                     #  r2_score(position_raw_x, rakf_state_estimation_x),
                     #  r2_score(position_raw_y, rakf_state_estimation_y),
                     #  r2_score(position_raw_z, rakf_state_estimation_z)],
                     # ["RAKF CLM ALGORITHM",
                     #  r2_score(position_raw_x, rakf_position_x),
                     #  r2_score(position_raw_y, rakf_position_y),
                     #  r2_score(position_raw_z, rakf_position_z)],
                     ["RAKF CVM ALGORITHM",
                      r2_score(position_raw_x, rakf_cvm_position_x),
                      r2_score(position_raw_y, rakf_cvm_position_y),
                      r2_score(position_raw_z, rakf_cvm_position_z)],
                     ["RAKF CAM ALGORITHM",
                      r2_score(position_raw_x, rakf_cam_position_x),
                      r2_score(position_raw_y, rakf_cam_position_y),
                      r2_score(position_raw_z, rakf_cam_position_z)]
                     ]
    print(tabulate(table_content, table_header, table_format))

    # MEAN SQUARE ERROR (MSE) TABLE
    table_format = "fancy_grid"
    table_header = ['MEAN SQUARE ERROR (MSE)',
                    'X-AXIS',
                    'Y-AXIS',
                    'Z-AXIS']
    table_content = [["RAW POSITION WITH OUTLIERS",
                      mean_squared_error(position_raw_x, position_raw_with_outlier_x),
                      mean_squared_error(position_raw_y, position_raw_with_outlier_y),
                      mean_squared_error(position_raw_z, position_raw_with_outlier_z)],
                     # ["KF ALGORITHM",
                     #  mean_squared_error(position_raw_x, kf_position_x),
                     #  mean_squared_error(position_raw_y, kf_position_y),
                     #  mean_squared_error(position_raw_z, kf_position_z)],
                     # ["STATE ESTIMATION ALGORITHM (WLS)",
                     #  mean_squared_error(position_raw_x, rakf_state_estimation_x),
                     #  mean_squared_error(position_raw_y, rakf_state_estimation_y),
                     #  mean_squared_error(position_raw_z, rakf_state_estimation_z)],
                     # ["RAKF CLM ALGORITHM",
                     #  mean_squared_error(position_raw_x, rakf_position_x),
                     #  mean_squared_error(position_raw_y, rakf_position_y),
                     #  mean_squared_error(position_raw_z, rakf_position_z)],
                     ["RAKF CVM ALGORITHM",
                      mean_squared_error(position_raw_x, rakf_cvm_position_x),
                      mean_squared_error(position_raw_y, rakf_cvm_position_y),
                      mean_squared_error(position_raw_z, rakf_cvm_position_z)],
                     ["RAKF CAM ALGORITHM",
                     mean_squared_error(position_raw_x, rakf_cam_position_x),
                     mean_squared_error(position_raw_y, rakf_cam_position_y),
                     mean_squared_error(position_raw_z, rakf_cam_position_z)]
                     ]
    print(tabulate(table_content, table_header, table_format))

    # MAX ERROR TABLE  = Maximum error from the true position
    table_format = "fancy_grid"
    table_header = ['MAX ERROR',
                    'X-AXIS',
                    'Y-AXIS',
                    'Z-AXIS']
    table_content = [["RAW POSITION WITH OUTLIERS",
                      max_error(position_raw_x, position_raw_with_outlier_x),
                      max_error(position_raw_y, position_raw_with_outlier_y),
                      max_error(position_raw_z, position_raw_with_outlier_z)],
                     # ["KF ALGORITHM",
                     #  max_error(position_raw_x, kf_position_x),
                     #  max_error(position_raw_y, kf_position_y),
                     #  max_error(position_raw_z, kf_position_z)],
                     # ["STATE ESTIMATION ALGORITHM (WLS)",
                     #  max_error(position_raw_x, rakf_state_estimation_x),
                     #  max_error(position_raw_y, rakf_state_estimation_y),
                     #  max_error(position_raw_z, rakf_state_estimation_z)],
                     # ["RAKF CLM ALGORITHM",
                     #  max_error(position_raw_x, rakf_position_x),
                     #  max_error(position_raw_y, rakf_position_y),
                     #  max_error(position_raw_z, rakf_position_z)],
                     ["RAKF CVM ALGORITHM",
                      max_error(position_raw_x, rakf_cvm_position_x),
                      max_error(position_raw_y, rakf_cvm_position_y),
                      max_error(position_raw_z, rakf_cvm_position_z)],
                     ["RAKF CAM ALGORITHM",
                      max_error(position_raw_x, rakf_cam_position_x),
                      max_error(position_raw_y, rakf_cam_position_y),
                      max_error(position_raw_z, rakf_cam_position_z)]
                     ]
    print(tabulate(table_content, table_header, table_format))
    #
    # # Execution Time table
    # table_format = "grid"
    # table_header = ['EXECUTION TIME ',
    #                 'XYZ axes (SECONDS)']
    # table_content = [["KF ALGORITHM",
    #                   numpy.mean(kf_exec_time)],
    #                  ["RAKF ALGORITHM",
    #                   numpy.mean(rakf_exec_time)]]
    # print(tabulate(table_content, table_header, table_format))

    # PLOTTING -------------------------------------------------------------------------
    ##
    plt.rcParams.update({'font.size': 20})
    plt.rcParams['lines.linewidth'] = 2

    # Individual axis plot
    fig1 = plt.figure()
    fig1plot1 = fig1.add_subplot(211)  # X axis
    fig1plot1.title.set_text("Random Walk X-axis")
    fig1plot1.set_xlabel("time(sec)")
    fig1plot1.set_ylabel("X position (m)")
    fig1plot1.plot(input_sample, position_raw_with_outlier_x, label="Measured position", color="r", linestyle=":",
                   marker=".", linewidth=0.8
                   )
    fig1plot1.plot(input_sample, position_raw_x, label="True position", color="g", linestyle="-")
    # fig1plot1.plot(input_sample, kf_position_x, label="kf", color="b", linestyle="dotted", marker=".")
    # fig1plot1.plot(input_sample, rakf_position_x, label="rakf clm", color="y", linestyle="-", marker=".")
    # fig1plot1.plot(input_sample, rakf_state_estimation_x, label="rakf wls", color="m", linestyle="dotted", marker=".")
    fig1plot1.plot(input_sample, rakf_cvm_position_x, label="RAKF CVM", color="k", linestyle="--")
    fig1plot1.plot(input_sample, rakf_cam_position_x, label="RAKF CAM", color="c", linestyle="-")
    plt.legend()

    fig1plot2 = fig1.add_subplot(212)   # Y axis
    fig1plot2.title.set_text("Random Walk Y-axis")
    fig1plot2.set_xlabel("time(sec)")
    fig1plot2.set_ylabel("Y position(m)")
    fig1plot2.plot(input_sample, position_raw_with_outlier_y, label="Measured position", color="r", linestyle=":",
                   marker=".",linewidth=1)
    fig1plot2.plot(input_sample, position_raw_y, label="True position", color="g", linestyle="-")
    # fig1plot2.plot(input_sample, kf_position_y, label="kf", color="b", linestyle="dotted", marker=".")
    # fig1plot2.plot(input_sample, rakf_position_y, label="rakf clm", color="y", linestyle="-", marker=".")
    # fig1plot2.plot(input_sample, rakf_state_estimation_y, label="rakf wls", color="m", linestyle="dotted", marker=".")
    fig1plot2.plot(input_sample, rakf_cvm_position_y, label="RAKF CVM", color="k", linestyle="--")
    fig1plot2.plot(input_sample, rakf_cam_position_y, label="RAKF CAM", color="c", linestyle="-")

    # ERROR PLOT
    fig2 = plt.figure()
    # X-AXIS
    fig2plot1 = fig2.add_subplot(211)
    fig2plot1.title.set_text("X-axis Error")
    fig2plot1.set_xlabel("time(sec)")
    fig2plot1.set_ylabel("X error(m)")
    fig2plot1.plot(input_sample, position_raw_with_outlier_error_x, label="Measured position error", color="r", linestyle=":",
                   marker=".", linewidth="1")
    # fig2plot1.plot(input_sample, kf_error_x, label="rakf error", color="b", linestyle="dotted", marker=".")  # KF
    # fig2plot1.plot(input_sample, rakf_error_x, label="rakf clm error", color="y", linestyle="dotted", marker=".")  # CLM
    fig2plot1.plot(input_sample, rakf_cvm_error_x, label="RAKF CVM Error", color="k", linestyle="--")
    fig2plot1.plot(input_sample, rakf_cam_error_x, label="RAKF CAM Error", color="c", linestyle="-")
    plt.legend()

    # Y-AXIS
    fig2plot1 = fig2.add_subplot(212)
    fig2plot1.title.set_text("Y-axis Error")
    fig2plot1.set_xlabel("time(sec)")
    fig2plot1.set_ylabel("Y error(m)")
    fig2plot1.plot(input_sample, position_raw_with_outlier_error_y, label="Measured position error", color="r", linestyle=":",
                   marker=".", linewidth=1)
    # fig2plot1.plot(input_sample, kf_error_y, label="kf error", color="b", linestyle="dotted", marker=".")
    # fig2plot1.plot(input_sample, rakf_error_y, label="rakf clm error", color="y", linestyle="dotted", marker=".")
    fig2plot1.plot(input_sample, rakf_cvm_error_y, label="RAKF CVM Error", color="k", linestyle="--")
    fig2plot1.plot(input_sample, rakf_cam_error_y, label="RAKF CAM Error", color="c", linestyle="-")

    # WALKING PATH PLOT
    fig3 = plt.figure()
    fig3plot1 = fig3.add_subplot(111)
    fig3plot1.title.set_text("Random Walk 2D")
    fig3plot1.set_xlabel("X position(m)")
    fig3plot1.set_ylabel("Y position(m)")
    fig3plot1.plot(position_raw_x, position_raw_y, label="True position", color="g", linestyle="-")    # TRUE
    fig3plot1.plot( position_raw_with_outlier_x, position_raw_with_outlier_y, label="Measured position", color="r",
    linestyle=":",marker='.',linewidth=0.8 )
    # fig3plot1.plot(kf_position_x, kf_position_y, label="kf", color="b", linestyle="dotted")   # KF
    # fig3plot1.plot(rakf_position_x, rakf_position_y, label="rakf clm", color="y", linestyle=":")    # RAKF CLM
    fig3plot1.plot(rakf_cvm_position_x, rakf_cvm_position_y, label="RAKF CVM", color="k", linestyle="--")
    fig3plot1.plot(rakf_cam_position_x, rakf_cam_position_y, label="RAKF CAM", color="c", linestyle="-")

    plt.legend()
    plt.show()
