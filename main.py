#Implementation of Deep Deterministic Gradient with Tensor Flow"
# Author: Steven Spielberg Pon Kumar (github.com/stevenpjg)

import gym
from gym.spaces import Box, Discrete
import numpy as np
from ddpg import DDPG
from ou_noise import OUNoise
import matplotlib as plt
#specify parameters here:
episodes=5000
is_batch_norm = True #batch normalization switch

def main():
    experiment= 'HRI_Mobile-v0' #specify environments here
    env= gym.make(experiment)
    steps= env.spec.timestep_limit #steps per episode    
    assert isinstance(env.observation_space, Box), "observation space must be continuous"
    assert isinstance(env.action_space, Box), "action space must be continuous"
    #Randomly initialize critic,actor,target critic, target actor network  and replay buffer   
    agent = DDPG(env, is_batch_norm)
    exploration_noise = OUNoise(env.action_space.shape[0], theta = 0.001, sigma = 0.002)
    counter=0
    reward_per_episode = 0    
    total_reward=0
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]    
    print ("Number of States:", num_states)
    print ("Number of Actions:", num_actions)
    print ("Number of Steps per episode:", steps)
    #saving reward:
    reward_st = np.array([0])
    rewards = []
    
    for i in range(episodes):
        print ("==== Starting episode no:",i,"====","\n")
        observation = env.reset()
        reward_per_episode = 0
        for t in range(steps):
            #rendering environmet (optional)            
            env.render()
            x = observation
            action = agent.evaluate_actor(np.reshape(x,[1,num_states]))
            #action = [np.pi]
            noise = exploration_noise.noise()
            action = action[0] + noise #Select action according to current policy and exploration noise

            print ("Observation at step", t, " :", observation, "\n")
            print ("Action at step", t ," :",action,"\n")

            observation,reward,done,info=env.step(action)
            
            #add s_t,s_t+1,action,reward to experience memory
            agent.add_experience(x,observation,action,reward,done)
            #train critic and actor network
            if counter > 64: 
                agent.train()
            reward_per_episode+=reward
            counter+=1
            #check if episode ends:
            if (done or (t == steps-1)):
                rewards.append(reward_per_episode)
                print ('EPISODE: ',i,' Steps: ',t,' Total Reward: ',reward_per_episode)
                print ("Printing reward to file")
                exploration_noise.reset() #reinitializing random noise for action exploration
                reward_st = np.append(reward_st,reward_per_episode)
                np.savetxt('episode_experiment.txt',reward_st, newline="\n")
                print ('\n\n')
                break
    total_reward+=reward_per_episode            
    print ("Average reward per episode {}".format(total_reward / episodes)  )
    print (rewards)
    x = [i for i in range(1, episodes+1)]
    fig = plt.figure()
    plt.plot(x, rewards)
    fig.savefig("~/Documents/research/HRI_Stationary_plots/reward_experiment.png")


if __name__ == '__main__':
    main()    