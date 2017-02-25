# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud

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
import gym
from gym import Envs, spaces
from gym.envs.registration import register
from gym.utils import seeding
import numpy as np
import random as rd
import math

class HRI_StationaryEnv(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        ####### Can I do this? #######
        self.max_action = np.array([1.4, 1.4]) # velocity
        self.min_action = np.array([-1.4, -1.4])
        self.min_position = -20.0
        self.max_position = 20.0
        self.speed = 2.0
        self.action = [0, 0]

        self.low_state = np.array([20, 20, 1.4, 1.4])
        self.high_state = np.array([-20.0, -20.0, -1.4, -1.4])

        self.viewer = None

        self.action_space = spaces.Box(self.min_action, self.max_action)
        self.observation_space = spaces.Box(self.low_state, self.high_state)

        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        position = self.position
        self.action = action

        for i in range(len(position)):
            position[i] = position[i] + action[i]

            if position[i] > self.max_position:
                position[i] = self.max_position
            elif position[i] < self.min_position:
                position[i] = self.min_position


        distance = np.linalg.norm(np.subtract(position, self.goal_position))
        """
        alpha = np.arctan((position[1]*np.cos(position[0]) + action[1]*np.cos(action[0]))/(position[1]*np.sin(position[0]) + action[1]*np.sin(action[0])))
        distance = np.sqrt((position[1]*np.cos(position[0]) + action[1]*np.cos(action[0]))**2 + (position[1]*np.sin(position[0]) + action[1]*np.sin(action[0]))**2)
        """
        done = bool(distance <= 10.0) #and np.linalg.norm(action) == 0)

        reward = 0.0
        if done:
            print ("#############")
            print ("DONE")
            print ("#############")
            reward = 10.0
        #print (self.distance)
        #print (self.position)
        #if self.distance > 20.0:
        #reward += 300.0*(2.0**(-np.sqrt(self.distance)))
        #else:
        reward -= 1.0
        #if done and np.action
        print ("Reward")
        print (reward)
        #if np.linalg.norm(self.distance - distance) <= 0.3:
        #reward -= float(self.step)*0.05
        self.distance = distance
        self.position = position
        self.state = np.array([self.goal_position[0]- position[0], self.goal_position[1] - position[1], action[0], action[1]])
        #self.state = np.array([position[0], position[1], action[0], action[1]])
        #self.step += 1
        return self.state, reward, done, {}

    def _reset(self):
        #x,y = np.random.uniform(self.min_position, self.max_position, 2)
        x1,y1 = np.random.uniform(10.0, self.max_position, 2)
        x1s, y1s = np.random.uniform(-1.0, 1.0, 2)
        #alpha = np.random.uniform(0.0, 2.0*np.pi)
        #distance = np.random.uniform(10.0, 20.0)
        self.position = [x1*x1s/abs(x1s), y1*y1s/abs(y1s)]
        #self.position = np.array([alpha, distance])
        self.goal_position = np.array([0.0, 0.0])
        self.distance = np.linalg.norm([x1*x1s/abs(x1s),y1*y1s/abs(y1s)])
        self.state = np.array([self.goal_position[0] - self.position[0], self.goal_position[1] - self.goal_position[1], self.action[0], self.action[1]])
        #self.state = np.array([alpha, distance, 0.0, 0.0])
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        #self.robtrans.set_translation((self.position[0]-self.min_position)*scale, (self.position[1]-self.min_position)*scale)
        #self.pedtrans.set_translation((self.goal_position[0]-self.min_position)*scale, (self.goal_position[1]-self.min_position)*scale)
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 400
        screen_height = 400

        world_width = self.max_position*2.0 # - self.min_position
        scale = screen_width/world_width

        if self.viewer is None:
            from gym.envs.RobotHumanInteraction import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.robtrans = rendering.Transform()
            pedestrian = rendering.make_circle(radius = 10)
            self.pedtrans = rendering.Transform()
            print (self.goal_position)
            pedestrian.add_attr(rendering.Transform(translation=(( self.goal_position[0]- self.min_position) *scale, (self.goal_position[0]-self.min_position)*scale)))
            self.robot = rendering.make_circle(radius = 10, filled= False)
            self.robot.add_attr(self.robtrans)
            self.viewer.add_geom(self.robot)
            self.viewer.add_geom(pedestrian)

        else:
            alpha = self.state[0]
            distance = self.state[1]
            #self.robtrans.set_translation(10.0, 10.0)
            self.robtrans.set_translation((self.position[0]-self.min_position)*scale, (self.position[1]-self.min_position)*scale)


        print (self.position)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

register(
    id='HRI_Stationary-v0',
    entry_point='gym.envs.RobotHumanInteraction:HRI_StationaryEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 15},
)
