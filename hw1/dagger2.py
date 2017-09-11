import numpy as np
import tensorflow as tf
import util, tf_util
import behavioral_cloning as policy
import load_policy
import gym
import pdb
import matplotlib.pyplot as plt

ENV = 'Walker2d-v1'
EXPERT_POLICY = 'experts/%s.pkl'%(ENV)
NUM_EPOCHS = 7#200#for each policy update
NUM_ITERATIONS = 10#40#num dagger iterations
NUM_ROLLOUTS = 20

def main():
  with tf.Session():
    tf_util.initialize()
    expert_policy = load_policy.load_policy(EXPERT_POLICY)
    
    summaries = []
    env = gym.make(ENV)
    data_mean, data_std = np.load('rollout_data/%s_standardize.npy'%(ENV))
    policy.STANDARDIZE = False
    x_train, x_cv, y_train, y_cv, = util.load(ENV)
    for i in range(NUM_ITERATIONS):
      train_agent(x_train, x_cv, y_train, y_cv, i)
      policy.load_dagger_model(ENV, i)#Unstandardized
      obs, mean, dev = rollout_policy(NUM_ROLLOUTS, env, data_mean, data_std)
      print('Iter: %d, Mean: %f, Dev: %f'%(i, mean, dev))
      actions = label_data(obs, expert_policy)
      x_train, x_cv, y_train, y_cv = aggregate(x_train, x_cv, y_train, y_cv, obs, actions)
      summaries.append([mean, dev])
  np.save('./dagger/%s/summaries.npy'%(ENV), summaries) 
  plot(np.array(summaries))

def train_agent(x_train, x_cv, y_train, y_cv, j):
  _, sess = policy.train_model([1e-4, .99, 1e-5, .7], x_train, x_cv, y_train, y_cv, NUM_EPOCHS, display=False)
  saver = saver = tf.train.Saver()
  saver.save(sess, './dagger/%s/model_%d.ckpt'%(ENV, j))
 
def rollout_policy(rollouts, env, mean, std):
  max_steps = env.spec.timestep_limit
  returns, means, devs = [], [], []
  observations = []
  i = 0
  for i in range(rollouts):
    obs = env.reset()
    done = False
    totalr = 0.
    steps = 0
    while not done:
      action = policy.predict(obs[None,:], mean, std)
      obs, r, done, _ = env.step(action)
      observations.append(obs)
      totalr += r
      steps += 1;
      if steps >= max_steps:
        break
    i += 1
    returns.append(totalr)
  return np.array(observations), np.mean(returns), np.std(returns)

def label_data(obs, expert_policy):
    actions = []
    for observation in obs:
      action = expert_policy(observation[None,:])
      actions.append(action[0])
    return np.array(actions)

def aggregate(x_train, x_cv, y_train, y_cv, observations, actions):
  x = np.append(x_train, observations, axis=0)
  y = np.append(y_train, actions, axis=0)
  return x,x_cv,y,y_cv

def plot(summaries):
  expert_mean, expert_std = np.load('./rollout_data/%s_performance.npy'%(ENV))
  means = summaries[:,0]
  devs = summaries[:,1]
  t = np.r_[:len(means)]+1
  plt.errorbar(t, means, devs, linestyle='None', marker='^')
  plt.plot(t, np.ones(len(t))*expert_mean, 'r-') #mean
  plt.xlabel('# Dagger Iterations')
  plt.ylabel('Return')
  plt.title('Avg return vs iteration of Dagger (%s)'%(ENV))
  plt.savefig('./dagger/%s/dagger_training.png'%(ENV))


if __name__ == '__main__':
  main()
