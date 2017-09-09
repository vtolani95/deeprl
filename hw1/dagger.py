import numpy as np
import tensorflow as tf
import util, tf_util
import bc_hopper as policy
import load_policy
import gym
import pdb

#envs = [['Hopper-v1', 10, 3
ENV = 'Hopper-v1'
EXPERT_POLICY = 'experts/%s.pkl'%(ENV)
NUM_EPOCHS = 200
NUM_ITERATIONS = 20

def main():
  summaries = []
  with tf.Session():
    expert_policy = load_policy.load_policy(EXPERT_POLICY)
    env = gym.make(ENV)
    x_train, x_cv, y_train, y_cv = util.load(ENV)
    policy.load_model('Hopper-v1_0.0001-0.99-1e-05')
    for i in range(NUM_ITERATIONS):
      #pdb.set_trace()
      policy.train_model([1e-4, .99, 1e-5], x_train, x_cv, y_train, y_cv, NUM_EPOCHS, display=False)
      obs, mean, dev = rollout_policy(1000, env)
      summaries.append([mean, dev])
      print('Mean: %f, Dev: %f'%(mean, dev))
      actions = label_data(obs, expert_policy)
      x_train, x_cv, y_train, y_cv = aggregate(x_train, x_cv, y_train, y_cv, obs, actions)
    plot(np.array(summaries))
 
def plot(summaries):
  means = summaries[:,0]
  devs = summaries[:,1]
  t = np.r_[:len(means)]+1
  plt.errorbar(t, means, devs, linestyle='None')
  plt.show()

def rollout_policy(num_examples, env):
  max_steps = env.spec.timestep_limit
  returns, means, devs = [], [], []
  observations = []
  i = 0
  while len(observations) < num_examples:
    #print('iter', i)
    obs = env.reset()
    done = False
    totalr = 0.
    steps = 0
    while not done:
      action = policy.predict(obs[None,:])
      observations.append(obs)
      obs, r, done, _ = env.step(action)
      totalr += r
      steps += 1;
      #if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
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
