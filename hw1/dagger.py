import numpy as np
import tensorflow as tf
import util, tf_util
import bc_ant as policy
import load_policy
import gym
import pdb
import matplotlib.pyplot as plt

#envs = [['Hopper-v1', 10, 3
ENV = 'Hopper-v1'
EXPERT_POLICY = 'experts/%s.pkl'%(ENV)
NUM_EPOCHS = 200#200#for each policy update
NUM_ITERATIONS = 20#40#num dagger iterations
NUM_EXAMPLES = 2000
def main():
  summaries = []
  with tf.Session():
    expert_policy = load_policy.load_policy(EXPERT_POLICY)
    env = gym.make(ENV)
    #mean, std = np.load('rollout_data/hopper_standardize.npy')
    x_train, x_cv, y_train, y_cv, mean_data, std_data = util.load(ENV)
    for i in range(NUM_ITERATIONS):
      #pdb.set_trace()
      policy.train_model([1e-3, .99, 1e-5, 1.0], x_train, x_cv, y_train, y_cv, NUM_EPOCHS, display=False)
      obs, mean, dev = rollout_policy(NUM_EXAMPLES, env, mean_data, std_data)
      summaries.append([mean, dev])
      print('Iter: %d, Mean: %f, Dev: %f'%(i, mean, dev))
      actions = label_data(obs, expert_policy)
      x_train, x_cv, y_train, y_cv = aggregate(x_train, x_cv, y_train, y_cv, obs, actions)
    np.save('./dagger_data/%s.npy'%(ENV), summaries) 
    plot(np.array(summaries))
 
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

def rollout_policy(num_examples, env, mean, std):
  max_steps = env.spec.timestep_limit
  returns, means, devs = [], [], []
  observations = []
  i = 0
  while len(observations) < num_examples:
    obs = env.reset()
    done = False
    totalr = 0.
    steps = 0
    while not done:
      action = policy.predict(obs[None,:], mean, std)
      observations.append(obs)
      obs, r, done, _ = env.step(action)
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
  ind = int(len(observations)*.8)
  observations, actions = util.shuffle_data(observations, actions)
  x_train = np.append(x_train, observations[:ind], axis=0)
  x_cv = np.append(x_cv, observations[ind:], axis=0)
  y_train = np.append(y_train, actions[:ind], axis=0)
  y_cv = np.append(y_cv, actions[ind:], axis=0)
  return x_train, x_cv, y_train, y_cv  

if __name__ == '__main__':
  main()
