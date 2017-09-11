import numpy as np
import tensorflow as tf
import util
import sys, os, pdb
import matplotlib.pyplot as plt

import bc_models.walker2d as bc_agent

ENV_NAME = bc_agent.ENV_NAME
VERSION = bc_agent.VERSION
STANDARDIZE = True

NUM_EXAMPLES = 16000
CV_SIZE = 4000
NUM_BATCHES_PER_EPOCH = 20
BATCH_SIZE = 50
NUM_EPOCHS = 400
NUM_EPOCHS_PER_DECAY = 5
DISPLAY_STEP = 10

def output_dir(params):
    dir_name = './tf/%s_v%s_%s'%(ENV_NAME, VERSION,"-".join([str(param) for param in params]))
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def train_epoch(x_train, y_train, hyperparam, sess):
    batch_loss, batch_acc = 0, 0
    for j in range(NUM_BATCHES_PER_EPOCH):
        ind = np.random.choice(NUM_EXAMPLES, BATCH_SIZE)
        x_batch, y_batch = x_train[ind], y_train[ind]
        _, curr_loss, curr_accuracy = sess.run([train, total_loss, accuracy], {x: x_batch,
                                                                  y: y_batch,
                                                                  init_learn_rate: hyperparam[0],
                                                                  decay_rate: hyperparam[1],
                                                                  reg: hyperparam[2],
                                                                  keep_prob: hyperparam[3]})
        batch_loss += curr_loss
        batch_acc += curr_accuracy
    return batch_loss/NUM_BATCHES_PER_EPOCH, batch_acc/NUM_BATCHES_PER_EPOCH

def cross_validate(x_cv, y_cv, hyperparam, sess):
    batch_loss, batch_acc = 0, 0
    num_batches = int(len(x_cv)/BATCH_SIZE)
    for i in range(num_batches):
        x_batch, y_batch = x_cv[i*BATCH_SIZE: (i+1)*BATCH_SIZE], y_cv[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
        curr_loss, curr_accuracy = sess.run([total_loss, accuracy], {x: x_batch,
                                                                        y: y_batch,
                                                                        init_learn_rate: hyperparam[0],
                                                                        decay_rate: hyperparam[1],
                                                                        reg: hyperparam[2],
                                                                        keep_prob: 1})
        batch_loss += curr_loss
        batch_acc += curr_accuracy
    return batch_loss/num_batches, batch_acc/num_batches

    
def loss(y, preds, reg, weights):
    weight_loss = tf.add_n([tf.norm(weight) for weight in weights.values()])
    pred_loss = tf.nn.l2_loss(y-preds)
    return pred_loss + reg*weight_loss

def predict(obs, mean, std):
    if STANDARDIZE:
      obs = util.standardize(obs, mean, std)
    actions = sess.run([preds], {x: obs, keep_prob: 1})
    return actions

def load_model(model, num):
    saver = tf.train.Saver()
    saver.restore(sess, './tf/%s/model_%d.ckpt'%(model, num))

def load_dagger_model(model, version, num):
    saver = tf.train.Saver()
    saver.restore(sess, './dagger/%s_%s/model_%d.ckpt'%(model, version, num))
#hyperparam- [learn rate, decay rate, l2 reg]
def train_model(hyperparam, x_train, x_cv, y_train, y_cv, num_epochs, display=True, save=False):
  print(util.green(str(hyperparam)))
  if save:
    saver = tf.train.Saver()
  init = tf.global_variables_initializer()
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
  sess.run(init)
  summaries = []
  for j in range(num_epochs):
    train_loss, train_acc = train_epoch(x_train, y_train, hyperparam, sess)
    cv_loss, cv_acc = cross_validate(x_cv, y_cv, hyperparam, sess)
    summaries.append([train_loss, train_acc, cv_loss, cv_acc])
    if display and j % DISPLAY_STEP == 0:
      print("Train Loss: %f\nCV Loss: %f\nTrain Accuracy: %s\nCV Accuracy: %s\nEPOCH: %d\n"%(train_loss,
                                                                    cv_loss,
                                                                    train_acc,
                                                                    cv_acc,
                                                                    j))
    if save and j % 50 == 0:
      saver.save(sess, output_dir(hyperparam) + '/model_%d.ckpt'%(j))
  if save:
    saver.save(sess, output_dir(hyperparam) + '/model_%d.ckpt'%(j))
  return np.array(summaries), sess

def plot_training(summaries, hyperparam):
    train_loss, train_acc, cv_loss, cv_acc = summaries[:,0], summaries[:,1], summaries[:,2], summaries[:,3]
    fig = plt.figure(1, figsize=(12, 10))
    fig.suptitle('%s: %s'%(ENV_NAME, str(hyperparam)))

    ax = fig.add_subplot(1,2,1)
    ax.plot(train_loss, 'r-', label='Train')
    ax.plot(cv_loss, 'b-', label='CV')
    plt.title('Loss Per Batch(50)')
    plt.xlabel('Epoch')
    plt.legend()

    ax = fig.add_subplot(1,2,2)
    ax.plot(train_acc, 'r-', label='Train')
    ax.plot(cv_acc, 'b-', label='CV')
    plt.title('Mean L2 Error Per Example(Over Batch[50])')
    plt.xlabel('Epoch')
    plt.legend()

global_step = tf.Variable(0, trainable=False)
init_learn_rate = tf.placeholder(tf.float32)
reg = tf.placeholder(tf.float32)
decay_rate = tf.placeholder(tf.float32)
learning_rate = tf.train.exponential_decay(init_learn_rate, global_step, NUM_EPOCHS_PER_DECAY*NUM_BATCHES_PER_EPOCH, decay_rate, staircase=True)
keep_prob = tf.placeholder(tf.float32)

x = tf.placeholder(tf.float32, shape=(None, bc_agent.NUM_OBS))
y = tf.placeholder(tf.float32, shape=(None, bc_agent.NUM_ACTIONS))

preds = bc_agent.model(x, keep_prob)
total_loss = loss(y, preds, reg, bc_agent.weights)
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

learn_rate_summary = tf.summary.scalar('learning_rate', learning_rate)

accuracy = tf.reduce_mean(tf.square(y-preds))
learning_rates = [1e-4]
decay_rates = [.99]
l2_regs = [1e-5]
dropouts = [1.0]
hyperparams = [[i, j, k, m] for i in learning_rates for j in decay_rates for k in l2_regs for m in dropouts]
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
if len(sys.argv) > 1 and sys.argv[1] == 'train':
    std = np.load('./rollout_data/%s_standardize.npy'%(ENV_NAME))
    if STANDARDIZE:
        x_train, x_cv, y_train, y_cv = util.load(ENV_NAME, True, std[0], std[1])
    else:
        x_train, x_cv, y_train, y_cv = util.load(ENV_NAME)
    for hyperparam in hyperparams:
        summaries, sess = train_model(hyperparam, x_train, x_cv, y_train, y_cv, NUM_EPOCHS)
        plot_training(summaries, hyperparam)
        plt.savefig(output_dir(hyperparam)+'/train_summary.png')
else:
    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(init)

