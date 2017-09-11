import numpy as np
import tensorflow as tf
import util, tf_util
import behavioral_cloning as policy
import load_policy
import gym
import pdb
import matplotlib.pyplot as plt

ENV = 'Reacher-v1'
EXPERT_POLICY = 'experts/%s.pkl'%(ENV)
NUM_EPOCHS = 100#200#for each policy update
NUM_ITERATIONS = 20#40#num dagger iterations
NUM_ROLLOUTS = 20

def main():
  summaries = []
  env = gym.make(ENV)
  data_mean, data_std = np.load('rollout_data/%s_standardize.npy'%(ENV))
  policy.STANDARDIZE = False
  if policy.STANDARDIZE:
    x_train, x_cv, y_train, y_cv, = util.load(ENV, True, data_mean, data_std)
  else:
    x_train, x_cv, y_train, y_cv, = util.load(ENV)
  x_train, x_cv, y_train, y_cv = x_train[:4000], x_cv[:1000], y_train[:4000], y_cv[:1000]
  for i in range(NUM_ITERATIONS):
#    saver = tf.train.Saver()
    #pdb.set_trace()
    train_agent(x_train, x_cv, y_train, y_cv, i)
    policy.load_dagger_model(ENV, i)#Unstandardized
    obs, mean, dev = rollout_policy(NUM_ROLLOUTS, env, data_mean, data_std)
    print('Iter: %d, Mean: %f, Dev: %f'%(i, mean, dev))
    actions = label_data(obs)
    x_train, x_cv, y_train, y_cv = aggregate(x_train, x_cv, y_train, y_cv, obs, actions)
#    summaries.append([mean, dev])
#  np.save('./dagger_data/%s.npy'%(ENV), summaries) 
#  plot(np.array(summaries))
def train_agent(x_train, x_cv, y_train, y_cv, j):
  _, sess = policy.train_model([1e-4, .99, 1e-5, 1.0], x_train, x_cv, y_train, y_cv, NUM_EPOCHS, display=False)
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

def label_data(obs):
  with tf.Session():
    expert_policy = load_policy.load_policy(EXPERT_POLICY)
    actions = []
    for observation in obs:
      action = expert_policy(observation[None,:])
      actions.append(action[0])
    return np.array(actions)

def aggregate(x_train, x_cv, y_train, y_cv, observations, actions):
  ind = int(len(observations)*.8)
  observations, actions = util.shuffle_data(observations, actions)
  x_train = np.append(x_train, observations[:ind], axis=0)
  x_cv = np.append(x_cv, observations[ind:], axis=0)
  y_train = np.append(y_train, actions[:ind], axis=0)
  y_cv = np.append(y_cv, actions[ind:], axis=0)
  return x_train, x_cv, y_train, y_cv  

def plot(summaries):
  means = summaries[:,0]
  devs = summaries[:,1]
  t = np.r_[:len(means)]+1
  plt.errorbar(t, means, devs, linestyle='None', marker='^')
  plt.plot(t, np.ones(len(t))*4814, 'r-') #mean
  plt.xlabel('# Dagger Iterations')
  plt.ylabel('Return')
  plt.title('Avg return vs iteration of Dagger')
  plt.savefig('./dagger_data/4_2_%s.png'%(ENV))


if __name__ == '__main__':
  main()
