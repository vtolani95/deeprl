#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import pdb
import behavioral_cloning as policy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    mean, std = np.load('rollout_data/%s_standardize.npy'%(args.envname))
    policy.ENV_NAME = args.envname
    policy.STANDARDIZE = True
    policy.load_model('Reacher-v1_v1.1_0.0001-0.99-1e-05-1.0', 399)#Unstandardized
    #policy.load_model('Ant-v1_v1.1_0.0001-0.99-1e-05-1.0', 399) #Stanardized
    print('loaded and built')


    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit
    
    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        #print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy.predict(obs[None,:], mean, std) 
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            #if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)
    expert_mean, expert_dev = np.load('./rollout_data/%s_performance.npy'%(args.envname))
    print('Expert Mean: %f, Expert Std: %f'%(expert_mean, expert_dev))
    print('BC Mean: %f, BC Std: %f'%(np.mean(returns), np.std(returns))) 

if __name__ == '__main__':
    main()
