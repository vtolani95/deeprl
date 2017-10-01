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

def plot_two():
  test1 = open('./data/test.txt', 'rb')
  test1 = test1.read()
  t1, mean1, best_mean1 = parse_log(test1)
  
  test3 = open('./data/test3.txt', 'rb')
  test3 = test3.read()
  t3, mean3, best_mean3 = parse_log(test3)
  
  test4 = open('./data/test4.txt', 'rb')
  test4 = test4.read()
  t4, mean4, best_mean4 = parse_log(test4)
  
  test5 = open('./data/test5.txt', 'rb')
  test5 = test5.read()
  t5, mean5, best_mean5 = parse_log(test5)
 
  plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
  plt.plot(t1, mean1, 'r-', label='1e-4')
  plt.plot(t3, mean3, 'b-', label='5e-4')
  plt.plot(t4, mean4, 'g-', label='3e-4')
  plt.plot(t5, mean5, 'm-', label='2e-4')
  plt.legend()
  plt.title('Mean Reward for Atari Pong with Different Initial Learning Rates')
  plt.xlabel('Timesteps')
  plt.ylabel('Reward')
  plt.show()

plot_one()
plot_two()
