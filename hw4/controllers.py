import numpy as np
from cost_functions import trajectory_cost_fn
import time
import pdb

class Controller():
  def __init__(self):
    pass

  # Get the appropriate action(s) for this state(s)
  def get_action(self, state):
    pass


class RandomController(Controller):
  def __init__(self, env):
    self.action_space = env.action_space
    self.cost_fn = None

  def get_action(self, state):
    """ Your code should randomly sample an action uniformly from the action space """
    return self.action_space.sample()


class MPCcontroller(Controller):
  """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
  def __init__(self,
         env,
         dyn_model,
         horizon=5,
         cost_fn=None,
         num_simulated_paths=10,
         ):
    self.env = env
    self.dyn_model = dyn_model
    self.horizon = horizon
    self.cost_fn = cost_fn
    self.num_simulated_paths = num_simulated_paths

  def get_action(self, state):
    ac_dim = self.env.action_space.shape[0]
    rand_acs = np.random.uniform(low=self.env.action_space.low,
                                high=self.env.action_space.high,
                                size=(self.num_simulated_paths, self.horizon, ac_dim))
   
    paths = []
    ob = np.ones((self.num_simulated_paths, len(state)))*state
    obs, acs, next_obs = [],[],[]
    for h in range(self.horizon):
        obs.append(ob)
        ac = rand_acs[:,h]
        ob = self.dyn_model.predict(ob, ac)
        next_obs.append(ob); acs.append(ac)
    obs, next_obs, acs = np.array(obs), np.array(next_obs), np.array(acs)
    costs = trajectory_cost_fn(self.cost_fn, obs, acs, next_obs)
    ind = np.argmin(costs)
    return acs[:,ind][0]
