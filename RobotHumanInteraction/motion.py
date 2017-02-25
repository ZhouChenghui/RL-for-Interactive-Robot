import numpy as np
import numpy.matlib


ObsDim = 3

class Motion:
    global ObsDim

    def __init__(_, typeOfMotion, initialState, sSqrtSigma, oSqrtSigma):
        _.Update = np.matlib.identity(3 * ObsDim)
        if typeOfMotion is "ConstantVelocity":
            for i in range(ObsDim, ObsDim*2):
                _.Update[i-ObsDim,i] = 1.0
        if typeOfMotion is "ConstantAcceleration":
            for i in range(ObsDim, ObsDim*3):
                _.Update[i-ObsDim,i] = 1.0
        _.Obs = np.concatenate((np.matlib.identity(ObsDim), np.zeros((ObsDim, ObsDim*2))), axis=1)
        _.initState = initialState
        _.states = [initialState]
        _.positions = [_.Obs*_.states[-1]]
        _.sqrtSigmaState = sSqrtSigma
        _.sqrtSigmaObs = oSqrtSigma
        _.motionType = typeOfMotion
        _.Obs = np.concatenate((np.matlib.identity(ObsDim), np.zeros((ObsDim, ObsDim*2))), axis=1)

    def noise(_, Ntype):

        if Ntype == 0: #state noise
            if _.motionType is "ConstantPosition":
                s = np.concatenate((np.random.randn(ObsDim, 1), np.zeros((6, 1))), axis = 0)

            if _.motionType is "ConstantVelocity":
                s = np.concatenate((np.random.randn(ObsDim*2, 1), np.zeros((3, 1))), axis = 0)

            if _.motionType is "ConstantAcceleration":
                s = np.random.randn(ObsDim*3, 1)
            s[2][0] = 0.0
            s[5][0] = 0.0
            s[8][0] = 0.0
            return _.sqrtSigmaState*(s)
        if Ntype == 1: #observation noise
            s = _.sqrtSigmaObs*np.random.randn(ObsDim, 1)
            s[2][0] = 0
            return s

    def step(_):
        _.states.append(_.Update * _.states[-1] + _.noise(0))
        _.positions.append(_.Obs * _.states[-1] + _.noise(1))


    def trajectory(_, steps):
        state = _.initState
        states = []
        observations = []
        for i in range(steps):
            obs = _.Obs*state + _.noise(1)
            state = _.Update*state + _.noise(0)
            observations.append(obs)
            states.append(state)
        return (states, observations)


    def get_position(_):
        return _.positions[-1]

    def get_state(_):
        return _.states[-1]


    def View(_):
        print("Position Relative to Robot: " + _.relativePosition[-1])



