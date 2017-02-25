import random as rd
import numpy as np
from gym.envs.RobotHumanInteraction.motion import Motion

def start_pt(rbt, r):
    pos = np.zeros((3,1))
    x = rd.uniform(-r, r)
    y = np.sqrt(r**2 - x**2)
    sign = rd.choice([1.0, -1.0])

    y = rbt[1] + y*sign
    x = rbt[0] + x

    pos[0, 0] = x
    pos[1, 0] = y
    pos[2, 0] = 0.0
    return pos

def start_v(v, dir):
    x = rd.uniform(-v, v)
    y = np.sqrt(v ** 2 - x ** 2)
    sign = rd.choice([1.0, -1.0])
    vel = np.zeros((3, 1))
    vel[0, 0] = x
    vel[1, 0] = sign * y
    while np.dot(dir.flatten(), vel.flatten()) <= 0.0:
        x = rd.uniform(-v, v)
        y = np.sqrt(v**2 - x**2)
        sign = rd.choice([1.0, -1.0])
        vel = np.zeros((3, 1))
        vel[0,0] = x
        vel[1,0] = sign*y
    return vel

class Robot(Motion):
    def __init__(_, motion_type, iniS, sSqrtSigma, oSqrtSigma, r):
        Motion.__init__(_,motion_type, iniS, sSqrtSigma, oSqrtSigma)
        _.ssrRadius = r

class Pedestrian(Motion):
    def __init__(_, robot_pos, motion_type, range):
        v = rd.uniform(0.7, 2.0)
        pt = start_pt(robot_pos, range)
        #_.robot = robot
        _.distance = range
        _.velocity = v
        sSqrtSigma = v / 30.0 * np.matlib.identity(9)
        oSqrtSigma = v / 30.0 * np.matlib.identity(3)

        if motion_type == "Approach":
            init = _.update_state_approach(pt, robot_pos)
            mt = "ConstantAcceleration"
            #mt = "ConstantVelocity"
        if motion_type == "Irrelevant":
            init = np.zeros((9, 1))
            for i in range(0, 3):
                init[i, 0] = pt[i, 0]
            vel = start_v(v, robot_pos - pt)
            for i in range(3, 6):
                init[i, 0] = vel[i-3, 0]
            mt = "ConstantVelocity"
        Motion.__init__(_, mt, init, sSqrtSigma, oSqrtSigma)
        _.walk = motion_type


    def step(_, rbt_p):
        state = _.get_state()
        
        if _.walk == "Approach":
            # give negative reward
            position = np.zeros((3, 1))
            for i in range(3):
                position[i, 0] = state[0:3, 0][i]
            state = _.update_state_approach(position, rbt_p)
            
            if np.linalg.norm(position - rbt_p) <= 1.0:
                print ("stop")
                for i in range(3, 9):
                    state[i, 0] = 0.0
                print (state)
        
        state = _.Update * state + _.noise(0)
        position = _.Obs * state + _.noise(1)
        _.states.append(state)
        _.positions.append(position)
        _.distance = np.linalg.norm(position - rbt_p)

    def update_state_approach(_, pedes_p, robot_p):
        print (pedes_p)
        print (robot_p)
        dire = robot_p - pedes_p
        l = np.linalg.norm(dire)
        state_new = np.zeros((9, 1))
        for i in range(0, 3):
            state_new[i, 0] = pedes_p[i, 0]
        for i in range(3, 6):
            state_new[i][0] = dire[i-3][0] * _.velocity / l
        for i in range(6, 9):
            
            if dire[i-6][0] == 0:
                state_new[i][0] = 0
            else:
                state_new[i][0] = -0.5 * state_new[i-3][0] ** 2 / dire[i-6][0]
            #state_new[i][0] = 0.0
        return state_new