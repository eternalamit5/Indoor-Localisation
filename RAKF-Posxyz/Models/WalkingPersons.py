import os
import matplotlib.pyplot as plt
import numpy
import yaml
from Algorithm.OutlierGenerator import OutlierGenerator
from Algorithm.WalkingPatternGenerator import WalkPatternGenerator

class WalkingPersons:
    def __init__(self, file_name):
        self.persons = []

        if os.path.exists( file_name ):
            with open( file_name, 'r' ) as json_file:
                # client_json = json.load( json_file )
                client_yaml = yaml.load( json_file, Loader=yaml.FullLoader )
                for item in client_yaml["walking-persons"]:
                    outlier_models = []
                    for outlier_model in item["outliers"]:
                        outlier_models.append( OutlierGenerator( mean=outlier_model["mean"], standard_deviation=outlier_model["standard-deviation"], number_of_outliers=outlier_model["number-of-outlier"],
                                                                 sample_size=outlier_model["sample-size"] ) )
                    self.persons.append({"id": item["id"],
                                         "model": WalkPatternGenerator( boundary=item["walker-attributes"]["walk-boundary"],
                                                                        avg_speed_mps=item["walker-attributes"]["max-walk-speed-mps"],
                                                                        walk_dimension=item["walker-attributes"]["walk-dimension"],
                                                                        outlier_model_x=outlier_models[0], outlier_model_y=outlier_models[1], outlier_model_z=outlier_models[2],
                                                                        mid_point=item["walker-attributes"]["sigmoid-attributes"]["mid-point"],
                                                                        steepness=item["walker-attributes"]["sigmoid-attributes"]["steepness"],
                                                                        angle_deviation_degrees=item["walker-attributes"]["sigmoid-attributes"]["angle-deviation-degrees"]
                                                                        )})

    def run(self, tdelta=-1):
        for person in self.persons:
            person["model"].update( tdelta=tdelta)

    def get_states(self, id):
        for person in self.persons:
            if id == person["id"]:
                return person["model"].get_states()


if __name__ == '__main__':
    n = 1000
    labourers = WalkingPersons( "../Config/WalkingPersons.yaml" )
    px = numpy.zeros( n )
    py = numpy.zeros( n )
    pz = numpy.zeros( n )
    ox = numpy.zeros( n )
    oy = numpy.zeros( n )
    oz = numpy.zeros( n )
    vx = numpy.zeros( n )
    vy = numpy.zeros( n )
    vz = numpy.zeros( n )
    ax = numpy.zeros( n )
    ay = numpy.zeros( n )
    az = numpy.zeros( n )
    jx = numpy.zeros( n )
    jy = numpy.zeros( n )
    jz = numpy.zeros( n )
    sample = numpy.zeros( n )
    for i in range( 1, n ):
        labourers.run(tdelta=0.7)
        states = labourers.get_states( "1234" )
        # update states
        px[i] = states["x_pos"]
        py[i] = states["y_pos"]
        pz[i] = states["z_pos"]
        ox[i] = states["x_outlier_pos"]
        oy[i] = states["y_outlier_pos"]
        oz[i] = states["z_outlier_pos"]
        vx[i] = states["x_velocity"]
        vy[i] = states["y_velocity"]
        vz[i] = states["z_velocity"]
        ax[i] = states["x_acceleration"]
        ay[i] = states["y_acceleration"]
        az[i] = states["z_acceleration"]
        jx[i] = states["x_jerk"]
        jy[i] = states["y_jerk"]
        jz[i] = states["z_jerk"]
        sample[i] = i

    fig1 = plt.figure()
    fig1plot1 = fig1.add_subplot( 311 )
    fig1plot1.title.set_text( "Random Walk x-axis" )
    fig1plot1.set_xlabel( "steps" )
    fig1plot1.set_ylabel( "position" )
    fig1plot1.plot( sample, ox, label="outlier", color="r", linestyle="-", marker="." )
    fig1plot1.plot( sample, px, label="actual", color="g", linestyle="--", marker="." )

    fig1plot2 = fig1.add_subplot( 312 )
    fig1plot2.title.set_text( "Random Walk y-axis" )
    fig1plot2.set_xlabel( "steps" )
    fig1plot2.set_ylabel( "position" )
    fig1plot2.plot( sample, oy, label="outlier", color="r", linestyle="-", marker="." )
    fig1plot2.plot( sample, py, label="actual", color="g", linestyle="--", marker="." )

    fig1plot3 = fig1.add_subplot( 313 )
    fig1plot3.title.set_text( "Random Walk z-axis" )
    fig1plot3.set_xlabel( "steps" )
    fig1plot3.set_ylabel( "position" )
    fig1plot3.plot( sample, oz, label="outlier", color="r", linestyle="-", marker="." )
    fig1plot3.plot( sample, pz, label="actual", color="g", linestyle="--", marker="." )

    fig2 = plt.figure()
    fig2plot1 = fig2.add_subplot( 111, projection='3d' )
    fig2plot1.title.set_text( "Random Walk 3D" )
    fig2plot1.set_xlabel( "x position" )
    fig2plot1.set_ylabel( "y position" )
    fig2plot1.set_zlabel( "z position" )
    fig2plot1.plot( ox, oy, oz, label="outlier", color="r", linestyle="--" )
    fig2plot1.plot( px, py, pz, label="actual", color="g", linestyle="--" )

    fig3 = plt.figure()
    fig3plot1 = fig3.add_subplot( 111 )
    fig3plot1.title.set_text( "Random Walk 2D" )
    fig3plot1.set_xlabel( "x position" )
    fig3plot1.set_ylabel( "y position" )
    fig3plot1.plot( ox, oy, label="outlier", color="r", linestyle="--" )
    fig3plot1.plot( px, py, label="actual", color="g", linestyle="--" )

    plt.legend()
    plt.show()
