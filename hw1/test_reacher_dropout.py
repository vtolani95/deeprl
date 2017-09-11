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
import bc_reacher
import matplotlib.pyplot as plt

def get_dropouts():
    import glob
    dirs = glob.glob('./tf/Reacher-v1_v1.1*')
    dirs = [x[5:] for x in dirs] #trim ./tf/
    dropouts = [float(x.split('-')[-1]) for x in dirs]
    return dirs, dropouts

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()
    dirs, dropouts = get_dropouts()
    print('loading and building expert policy')
    print('loaded and built')
    model_means, model_devs = [], []
    for i in range(len(dirs)):
        bc_reacher.load_model(dirs[i])
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit
        means, devs = [], []
        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = bc_reacher.predict(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)
            means.append(np.mean(returns))
            devs.append(np.std(returns))
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
        model_means.append(np.mean(means))
        model_devs.append(np.mean(devs))
    #pdb.set_trace()
    fig = plt.figure(1, figsize=(8, 12))
    fig.clf()
    ax = fig.add_subplot(1,1,1)
    ax.errorbar(dropouts, model_means, model_devs, linestyle='None', marker='^')
    ax.plot(dropouts, np.ones(len(dropouts))*-3.9318, 'r-')
    plt.title('Mean and Std of Reacher Return vs Keep Prob')
    plt.xlabel('Keep Prob')
    plt.ylabel('Return')
#    ax = fig.add_subplot(2, 1, 1)
   # ax.stem(dropouts, model_means)
 #   plt.title('Mean Return')
  #  plt.xlabel('Training Dropout Percentage')
    
    #ax = fig.add_subplot(2,1,2)
    #ax.stem(dropouts, model_devs)
    #plt.title('Std Dev Return')
    #plt.xlabel('Training Dropout Percentage')
    plt.savefig('2_2.png')
        #expert_data = {'observations': np.array(observations),
                  #     'actions': np.array(actions)}
        #pdb.set_trace()
        #np.save('./rollout_data/%s'%(args.envname), expert_data)
if __name__ == '__main__':
    main()
