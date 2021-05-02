import matplotlib.pyplot as plt
import numpy
from personnel.tracking.position_node import create_trackers
from personnel.motion.walk_gen import create_walkers
from sklearn.metrics import r2_score, max_error, mean_squared_error
from tabulate import tabulate
import logging

logging.basicConfig(level=logging.WARNING, format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

if __name__ == '__main__':
    sample_time = 0.1
    sample_timestamp = sample_time

    number_of_samples = 100

    rakf_position_x = numpy.zeros(number_of_samples)
    rakf_position_y = numpy.zeros(number_of_samples)
    rakf_position_z = numpy.zeros(number_of_samples)

    rakf_gain_x = numpy.zeros(number_of_samples)
    # rakf_gain_y = numpy.zeros( number_of_samples )
    # rakf_gain_z = numpy.zeros( number_of_samples )

    position_raw_x = numpy.zeros(number_of_samples)
    position_raw_y = numpy.zeros(number_of_samples)
    position_raw_z = numpy.zeros(number_of_samples)

    position_error_x = numpy.zeros(number_of_samples)
    position_error_y = numpy.zeros(number_of_samples)
    position_error_z = numpy.zeros(number_of_samples)

    adaptive_x = numpy.zeros(number_of_samples)
    # adaptive_y = numpy.zeros( number_of_samples )
    # adaptive_z = numpy.zeros( number_of_samples )

    position_raw_with_outlier_x = numpy.zeros(number_of_samples)
    position_raw_with_outlier_y = numpy.zeros(number_of_samples)
    position_raw_with_outlier_z = numpy.zeros(number_of_samples)

    position_error_raw_meas_x = numpy.zeros(number_of_samples)
    position_error_raw_meas_y = numpy.zeros(number_of_samples)
    position_error_raw_meas_z = numpy.zeros(number_of_samples)

    velocity_raw_x = numpy.zeros(number_of_samples)
    # velocity_raw_y = numpy.zeros( number_of_samples )
    # velocity_raw_z = numpy.zeros( number_of_samples )
    #
    acceleration_raw_with_outlier_x = numpy.zeros(number_of_samples)
    # acceleration_raw_with_outlier_y = numpy.zeros( number_of_samples )
    # acceleration_raw_with_outlier_z = numpy.zeros( number_of_samples )
    #
    # jerk_raw_with_outlier_x = numpy.zeros( number_of_samples )
    # jerk_raw_with_outlier_y = numpy.zeros( number_of_samples )
    # jerk_raw_with_outlier_z = numpy.zeros( number_of_samples )

    input_sample = numpy.zeros(number_of_samples)

    walkers = create_walkers()
    tracker_rakf = create_trackers()

    for i in range(1, number_of_samples):
        states = walkers[0]["model"].update(sample_time)
        data = dict(measurement_x=states["x_outlier_pos"],
                    measurement_y=states["y_outlier_pos"],
                    measurement_z=states["z_outlier_pos"],
                    timestamp_ms=sample_timestamp,
                    velocity_x=states["x_velocity"],
                    velocity_y=states["y_velocity"],
                    velocity_z=states["z_velocity"],
                    acceleration_x=0,
                    acceleration_y=0,
                    acceleration_z=0)  # states["z_acceleration"] , states["z_velocity"]

        # data = dict(measurement_x=states["x_outlier_pos"],
        #            measurement_y=states["y_outlier_pos"],
        #            measurement_z=states["z_outlier_pos"],
        #            timestamp_ms=sample_timestamp,
        #            velocity_x=states["x_velocity"],
        #            velocity_y=states["y_velocity"],
        #            velocity_z=states["z_velocity"],
        #            acceleration_x=states["x_acceleration"],
        #            acceleration_y=states["y_acceleration"],
        #            acceleration_z=states["z_acceleration"])  # states["z_acceleration"] , states["z_velocity"]

        result = tracker_rakf[0]["model"].run(data)
        # result = tracker_rakf[0]["model"].run_on_measurements( data=data )

        # update sample_timestamp
        sample_timestamp = sample_timestamp + sample_time

        # update states
        rakf_position_x[i] = result[0]["state_model"]
        rakf_position_y[i] = result[1]["state_model"]
        rakf_position_z[i] = result[2]["state_model"]

        # rakf_gain_x[i] = result[0]["delta_state_estimate"]
        # rakf_gain_y[i] = result[1]["delta_state_estimate"]
        # rakf_gain_z[i] = result[2]["delta_state_estimate"]

        position_raw_x[i] = states["x_pos"]
        position_raw_y[i] = states["y_pos"]
        position_raw_z[i] = states["z_pos"]

        position_raw_with_outlier_x[i] = states["x_outlier_pos"]
        position_raw_with_outlier_y[i] = states["y_outlier_pos"]
        position_raw_with_outlier_z[i] = states["z_outlier_pos"]

        velocity_raw_x[i] = states["x_velocity"]
        # velocity_raw_y[i] = states["y_velocity"]
        # velocity_raw_z[i] = states["z_velocity"]
        #
        acceleration_raw_with_outlier_x[i] = states["x_acceleration"]
        # acceleration_raw_with_outlier_y[i] = states["y_acceleration"]
        # acceleration_raw_with_outlier_z[i] = states["z_acceleration"]
        #
        # jerk_raw_with_outlier_x[i] = states["x_jerk"]
        # jerk_raw_with_outlier_y[i] = states["y_jerk"]
        # jerk_raw_with_outlier_z[i] = states["z_jerk"]

        position_error_x[i] = position_raw_x[i] - rakf_position_x[i]
        position_error_y[i] = position_raw_y[i] - rakf_position_y[i]
        position_error_z[i] = position_raw_z[i] - rakf_position_z[i]

        position_error_raw_meas_x[i] = position_raw_x[i] - position_raw_with_outlier_x[i]
        position_error_raw_meas_y[i] = position_raw_y[i] - position_raw_with_outlier_y[i]
        position_error_raw_meas_z[i] = position_raw_z[i] - position_raw_with_outlier_z[i]

        adaptive_x[i] = result[4]["eqn38"]
        # adaptive_y[i] = result[4]["eqn38"]
        # adaptive_z[i] = result[5]["eqn38"]

        input_sample[i] = i
        print(i)

    ##
    plt.rcParams.update({'font.size': 16})

    # INDIVIDUAL AXIS PLOT
    fig1 = plt.figure()
    fig1plot1 = fig1.add_subplot(211)
    fig1plot1.title.set_text("Random Walk x-axis")
    fig1plot1.set_xlabel("steps")
    fig1plot1.set_ylabel("position")
    fig1plot1.plot(input_sample, position_raw_with_outlier_x, label="outlier", color="r", linestyle="dotted",
                   marker=".")
    fig1plot1.plot(input_sample, position_raw_x, label="actual", color="g", linestyle="-")
    fig1plot1.plot(input_sample, rakf_position_x, label="rakf", color="y", linestyle="-")

    fig1plot2 = fig1.add_subplot(212)
    fig1plot2.title.set_text("Random Walk y-axis")
    fig1plot2.set_xlabel("steps")
    fig1plot2.set_ylabel("position")
    fig1plot2.plot(input_sample, position_raw_with_outlier_y, label="outlier", color="r", linestyle="dotted",
                   marker=".")
    fig1plot2.plot(input_sample, position_raw_y, label="actual", color="g", linestyle="-")
    fig1plot2.plot(input_sample, rakf_position_y, label="rakf", color="y", linestyle="-")

    # WALK PATTERN PLOT
    fig3 = plt.figure()
    fig3plot1 = fig3.add_subplot(111)
    fig3plot1.set_title("Random Walk 2D")
    fig3plot1.set_xlabel("X position")
    fig3plot1.set_ylabel("Y position")
    fig3plot1.plot(position_raw_with_outlier_x, position_raw_with_outlier_y, label="outlier", color="r",
                   linestyle="dotted")
    fig3plot1.plot(position_raw_x, position_raw_y, label="actual", color="g", linestyle="-")
    fig3plot1.plot(rakf_position_x, rakf_position_y, label="rakf", color="y", linestyle="-")
    fig3plot1.legend(['Measured position (UWB)', 'True position'])

    # ERROR PLOT
    fig4 = plt.figure()
    fig4plot1 = fig4.add_subplot(211)
    fig4plot1.title.set_text("Error Plot")
    fig4plot1.set_xlabel("Sample")
    fig4plot1.set_ylabel("Error")
    # fig3plot1.plot( position_raw_with_outlier_x, position_raw_with_outlier_y, label="outlier", color="r", linestyle="--" )
    fig4plot1.plot(input_sample, position_error_raw_meas_x, label="raw_pos_error_x", color="y", linestyle="-")
    # fig4plot1.plot( input_sample, position_error_raw_meas_y, label="raw_pos_error_y", color="g", linestyle="-" )
    fig4plot1.plot(input_sample, position_error_x, label="pos_error_x", color="k", linestyle="-")
    # fig4plot1.plot( input_sample, position_error_y, label="pos_error_y", color="m", linestyle="-" )

    # fig4 = plt.figure()
    fig4plot1 = fig4.add_subplot(212)
    fig4plot1.title.set_text("Error Plot")
    fig4plot1.set_xlabel("Sample")
    fig4plot1.set_ylabel("Error")
    # fig3plot1.plot( position_raw_with_outlier_x, position_raw_with_outlier_y, label="outlier", color="r", linestyle="--" )
    # fig4plot1.plot( input_sample, position_error_raw_meas_x, label="raw_pos_error_x", color="y", linestyle="-" )
    fig4plot1.plot(input_sample, position_error_raw_meas_y, label="raw_pos_error_y", color="g", linestyle="-")
    # fig4plot1.plot( input_sample, position_error_x, label="pos_error_x", color="k", linestyle="-" )
    fig4plot1.plot(input_sample, position_error_y, label="pos_error_y", color="m", linestyle="-")

    # fig5 = plt.figure()
    # fig5plot1 = fig5.add_subplot( 212 )
    # fig5plot1.title.set_text( "Adaptive factor Plot" )
    # fig5plot1.set_xlabel( "Sample" )
    # fig5plot1.set_ylabel( "Adaptive factor" )

    # # TABLE -------------------------------------------------------------------------------------
    table_format = "fancy_grid"
    table_header = ['METRICS',
                    'X-AXIS',
                    'Y-AXIS',
                    'Z-AXIS']
    table_content = [
        ["RAW position Error variance",
         numpy.std(position_error_raw_meas_x),
         numpy.std(position_error_raw_meas_y),
         numpy.std(position_error_raw_meas_z)],
        ["RAKF Error variance",
         numpy.std(position_error_x),
         numpy.std(position_error_y),
         numpy.std(position_error_z)],
        ["RAW POSITION WITH OUTLIERS R^2",
         r2_score(position_raw_x, position_raw_with_outlier_x),
         r2_score(position_raw_y, position_raw_with_outlier_y),
         r2_score(position_raw_z, position_raw_with_outlier_z)],
        ["RAKF ALGORITHM R^2 ",
         r2_score(position_raw_x, rakf_position_x),
         r2_score(position_raw_y, rakf_position_y),
         r2_score(position_raw_z, rakf_position_z)]
    ]
    print(tabulate(table_content, table_header, table_format))

    plt.legend()
    plt.show()
