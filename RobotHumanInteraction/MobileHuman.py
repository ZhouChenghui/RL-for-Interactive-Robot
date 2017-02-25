# -*- coding: utf-8 -*-
"""
@author: Chenghui Zhou

A merge between two sources:

* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia

* the OpenAI/gym MountainCar environment
itself from
https://webdocs.cs.ualberta.ca/~sutton/MountainCar/MountainCar1.cp
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
import random as rd
import math
from RobotHumanInteraction.sceneComponents import Pedestrian

class HRI_MobileEnv(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.range = 40.0
        self.diameter = 50.0

        self.max_action = [math.pi, 2.0] # velocity
        self.min_action = [-math.pi, 0.0]

        self.min_position = [-math.pi, 0.0]
        self.max_position = [math.pi, self.diameter]

        self.sensor_range = 10.0

        self.low_state = np.array([self.min_position[0], self.min_position[1], self.min_action[0], self.max_action[1]])
        self.high_state = np.array([self.max_position[0], self.max_position[1], self.max_action[0], self.max_action[1]])

        self.viewer = None

        self.action_space = spaces.Box(np.array(self.min_action), np.array(self.max_action))
        self.observation_space = spaces.Box(self.low_state, self.high_state)

        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        """Update robot xy position"""
        pos_xy = self.position_xy
        y = pos_xy[1] + action[1]*np.sin(action[0])
        x = pos_xy[0] + action[1]*np.cos(action[0])
        self.position_xy = np.array([x, y])

        """Update robot polar position"""
        alpha = np.arctan2(y, x)
        distance = np.sqrt(x**2 + y**2)
        self.position = np.array([alpha, distance])

        """Update pedestrian position"""
        self.pedestrian.step(np.matrix([[x], [y], [0.0]]))
        p = self.pedestrian.get_position()
        self.goal_position_xy = np.array([p[0, 0], p[1, 0]])

        """Calculate relative position"""
        rel_y = self.goal_position_xy[1] - y
        rel_x = self.goal_position_xy[0] - x
        rel_alpha = np.arctan2(rel_y, rel_x)
        rel_distance = np.sqrt(rel_x**2 + rel_y**2)

        done = bool(rel_distance <= 2.0 and abs(action[1]) < 0.1) # and np.linalg.norm(action) == 0)

        reward = 0.0
        if done:
            print ("#############")
            print ("DONE")
            print ("#############")
            #reward = 0.0

        reward -= 10.0
        print ("Reward")
        print (reward)
        self.state = np.array([rel_alpha, rel_distance, action[0], action[1]])

        return self.state, reward, done, {}

    def _reset(self):

        alpha = np.random.uniform(-np.pi, np.pi)
        distance = np.random.uniform(0.0, 50.0)

        """Initialize robot position"""
        self.position = np.array([alpha, distance])
        self.position_xy = np.array([distance*np.cos(alpha), distance*np.sin(alpha)])

        """Initialize pedestrian postiion"""
        p_robot = self.position_xy
        self.pedestrian= Pedestrian(np.matrix([[p_robot[0]], [p_robot[1]], [0.0]]), "Approach", self.range)
        p = self.pedestrian.get_position()

        self.goal_position_xy = np.array([p[0, 0], p[1, 0]])
        p_goal = self.goal_position_xy
        """Initialize relative position/states"""
        rel_x = p_goal[0] - p_robot[0]
        rel_y = p_goal[1] - p_robot[1]
        self.state = np.array([np.arctan2(rel_y, rel_x), np.sqrt(rel_x**2 + rel_y**2), 0.0, 0.0])

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 600

        world_width = self.max_position[1]*2.0 # - self.min_position
        scale = screen_width/world_width
        min = -self.max_position[1]

        if self.viewer is None:
            from gym.envs.RobotHumanInteraction import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.robtrans = rendering.Transform()
            self.pedtrans = rendering.Transform()
            self.pedes = rendering.make_circle(radius = 10)
            self.robot = rendering.make_circle(radius = 10, filled= False)
            self.pedes.add_attr(self.pedtrans)
            self.robot.add_attr(self.robtrans)
            self.viewer.add_geom(self.robot)
            self.viewer.add_geom(self.pedes)
            self.robtrans.set_translation((self.position_xy[0]-min)*scale, (self.position_xy[1]-min)*scale)
            self.pedtrans.set_translation((self.goal_position_xy[0]-min)*scale, (self.goal_position_xy[1]-min)*scale)
        else:
            self.robtrans.set_translation((self.position_xy[0]-min)*scale, (self.position_xy[1]-min)*scale)
            self.pedtrans.set_translation((self.goal_position_xy[0]-min)*scale, (self.goal_position_xy[1]-min)*scale)

        #print (self.position)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

register(
    id='HRI_Mobile-v0',
    entry_point='RobotHumanInteraction:HRI_MobileEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 50},
)
