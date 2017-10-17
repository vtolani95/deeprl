import tensorflow as tf
import numpy as np
import pdb

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ Note: Be careful about normalization """
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.sess = sess
        self.env = env

        self.mean_obs, self.std_obs, self.mean_deltas, self.std_deltas, self.mean_action, self.std_action = normalization
        ob_dim, ac_dim = env.observation_space.shape[0], env.action_space.shape[0]
        #Placeholders for normalized inputs, and outputs
        self.acs = tf.placeholder(shape=[None, ac_dim],
                                name='action',
                                dtype=tf.float32)
        self.obs = tf.placeholder(shape=[None, ob_dim],
                                name='obs',
                                dtype=tf.float32)
        self.deltas= tf.placeholder(shape=[None, ob_dim],
                                name='deltas',
                                dtype=tf.float32)
        #normalization
        eps = 1e-7
        self.a_hat = (self.acs-self.mean_action)/(self.std_action+eps)
        self.ob_hat = (self.obs-self.mean_obs)/(self.std_obs+eps)
        
        #MLP 
        self.mlp_input = tf.concat((self.a_hat, self.ob_hat), axis=1)
        self.mlp_output = build_mlp(input_placeholder=self.mlp_input,
                            output_size=ob_dim,
                            scope='dynamics',
                            n_layers=n_layers,
                            size=size,
                            activation=activation,
                            output_activation=output_activation)
          
        self.normalized_output = self.mlp_output*self.std_deltas + self.mean_deltas
        #Loss and Train Op
        self.loss = tf.nn.l2_loss(self.normalized_output-self.deltas, name='dynamics_loss')
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss) 
    

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        #extract data
        obs = np.vstack([path['observations'] for path in data])
        next_obs = np.vstack([path['next_observations'] for path in data])
        actions = np.vstack([path['actions'] for path in data])
        deltas = next_obs - obs
        
        for i in range(self.iterations):
            ind = np.random.choice(len(obs), self.batch_size)
            ob_batch, a_batch, delta_batch = obs[ind], actions[ind], deltas[ind]
            _, loss = self.sess.run((self.train_op, self.loss), {self.obs: ob_batch,
                                              self.acs: a_batch,
                                              self.deltas: delta_batch})

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        deltas = self.sess.run(self.normalized_output, {self.obs: states, self.acs: actions}) 
        return states + deltas
