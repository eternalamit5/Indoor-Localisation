import matplotlib.pyplot as plt
import pandas as pd
import time

fig = plt.figure()
ax1 = fig.add_subplot( 1, 1, 1 )

plot_filtered_pos_x = []
plot_filtered_pos_y = []
plot_raw_pos_x = []
plot_raw_pos_y = []


def readData(csv_file_name):
    # Step1: Read the Data from sensor
    # or
    # Step1: Read the Data from csv
    df = pd.read_csv( csv_file_name )

    # Step2: Pass the measured data to the algorithm
    for index, row in df.iterrows():
        plot_filtered_pos_x.append( float( row['x_f'] ) )
        plot_filtered_pos_y.append( float( row['y_f'] ) )
        plot_raw_pos_x.append( float( row['x_r'] ) )
        plot_raw_pos_y.append( float( row['y_r'] ) )
        time.sleep( 0.1 )


def animate(i):
    ax1.clear()
    ax1.plot( plot_raw_pos_x, plot_raw_pos_y(),
              color='red' )
    ax1.scatter( plot_filtered_pos_x, plot_filtered_pos_y, color='blue',
                 facecolors='none' )
