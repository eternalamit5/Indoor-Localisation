import numpy as np


class ParameterEstimation:
    def __init__(self, sampleSize):
        # Parameter estimation variables
        self.sampleSize = sampleSize
        self.Mmat = np.zeros( (sampleSize, 1) )
        self.YBuffer = np.zeros( (1, sampleSize) )
        self.PbarBuffer = np.zeros( (1, sampleSize) )

    def estimate(self, meas, Pbar):
        # rolling adding new measurement to YBuffer
        self.YBuffer = np.roll( self.YBuffer, -1 )  # YBuffer shifted one step to the left
        self.YBuffer[0, self.sampleSize - 1] = meas  # Last position of YBbuffer is added with the new measurement

        # Parameter calculation
        MmatWeighted = np.multiply( np.transpose( self.Mmat ), self.PbarBuffer )  # Calculating "Mmatweighted" matrix
        # print(MmatWeighted)
        MtWY = np.matmul( MmatWeighted, np.transpose( self.YBuffer ) )
        MtWM = np.matmul( MmatWeighted, self.Mmat )

        # print(MtWM)

        param = (1/MtWM) * MtWY  # calculating the parameter
        # print(param)

        self.Mmat = np.roll( self.Mmat, -1, axis=0 )  # shifting the "Mmat" matrix up
        self.Mmat[self.sampleSize - 1, 0] = meas  # Last row of "Mmat" is added with new measurement

        self.PbarBuffer = np.roll( self.PbarBuffer, -1 )
        self.PbarBuffer[0, self.sampleSize - 1] = Pbar

        estimatedPosition = self.Mmat * param
        # print(estimatedPosition[0, 0])

        return estimatedPosition[0, 0]