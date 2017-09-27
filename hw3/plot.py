import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pdb


def parse_log(txt):
  txt = txt.decode('utf-8')
  lines = txt.split('\n')
  timesteps, mean, best_mean = [], [], []
  for elem in lines:
    if 'Timestep' in elem:
      elem = elem.split(' ')
      timesteps.append(int(elem[-1]))
    if 'mean reward (100 episodes)' in elem:
      elem = elem.split(' ')
      mean.append(float(elem[-1]))
    if 'best mean reward' in elem:
      elem = elem.split(' ')
      best_mean.append(float(elem[-1])) 
  return timesteps, mean, best_mean 

def plot_one():
  txt = open('./data/test.txt', 'rb')
  txt = txt.read()
  t, mean, best_mean = parse_log(txt)
  plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
  plt.plot(t, mean, 'r-', label='Mean Reward')
  plt.plot(t, best_mean, 'b-', label='Best Mean Reward')
  plt.legend()
  plt.title('Learning Curve for Atari Pong')
  plt.xlabel('Timesteps')
  plt.ylabel('Reward')
  plt.show()

plot_one()
