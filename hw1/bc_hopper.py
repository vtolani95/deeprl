import numpy as np
import tensorflow as tf
import util
import sys, os, pdb
import matplotlib.pyplot as plt

NUM_ACTIONS = 3
NUM_OBS = 11
ENV_NAME = 'Hopper-v1'
VERSION = '1.1'
NUM_EXAMPLES = 16000
CV_SIZE = 4000
NUM_BATCHES_PER_EPOCH = 20
BATCH_SIZE = 50
NUM_EPOCHS = 400
NUM_EPOCHS_PER_DECAY = 5
FC1_SIZE = 10
FC2_SIZE = 10
FC3_SIZE = 7
DISPLAY_STEP = 10

def output_dir(params):
    dir_name = './tf/%s_v%s_%s'%(ENV_NAME, VERSION,"-".join([str(param) for param in params]))
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def model(input_layer, weights, biases, keep_prob):
    fc1 = tf.nn.relu(tf.matmul(input_layer, weights['fc1']+biases['b1']))
    fc2 = tf.nn.relu(tf.matmul(fc1, weights['fc2']+biases['b2']))
    fc3 = tf.nn.relu(tf.matmul(fc1, weights['fc3']+biases['b3']))
    dropout = tf.nn.dropout(fc3, keep_prob)
    preds = tf.matmul(dropout, weights['fc4'] + biases['b4'])
    return preds    

def train_epoch(x_train, y_train, hyperparam, prob, sess):
    batch_loss, batch_acc = 0, 0
    for j in range(NUM_BATCHES_PER_EPOCH):
        ind = np.random.choice(NUM_EXAMPLES, BATCH_SIZE)
        x_batch, y_batch = x_train[ind], y_train[ind]
        _, curr_loss, curr_accuracy = sess.run([train, total_loss, accuracy], {x: x_batch,
                                                                  y: y_batch,
                                                                  init_learn_rate: hyperparam[0],
                                                                  decay_rate: hyperparam[1],
                                                                  reg: hyperparam[2],
                                                                  keep_prob: prob})
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

    
def loss(y, preds, reg):
    weight_loss = tf.add_n([tf.norm(weight) for weight in weights.values()])
    pred_loss = tf.nn.l2_loss(y-preds)
    return pred_loss + reg*weight_loss


def predict(obs):
    actions = sess.run([preds], {x: obs, keep_prob: 1})
    return actions

def load_model(model):
    saver = tf.train.Saver()
    saver.restore(sess, './tf/%s/model.ckpt'%(model))

#hyperparam- [learn rate, decay rate, l2 reg]
def train_model(hyperparam, x_train, x_cv, y_train, y_cv, num_epochs, display=True):
  print(util.green(str(hyperparam)))
  saver = tf.train.Saver()
  init = tf.global_variables_initializer()
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
  sess.run(init)
  summaries = []
  for j in range(num_epochs):
    train_loss, train_acc = train_epoch(x_train, y_train, hyperparam, 1, sess)
    cv_loss, cv_acc = cross_validate(x_cv, y_cv, hyperparam, sess)
    summaries.append([train_loss, train_acc, cv_loss, cv_acc])
    if display and j % DISPLAY_STEP == 0:
      print("Train Loss: %f\nCV Loss: %f\nTrain Accuracy: %s\nCV Accuracy: %s\nEPOCH: %d\n"%(train_loss,
                                                                    cv_loss,
                                                                    train_acc,
                                                                    cv_acc,
                                                                    j))
  return np.array(summaries)

def plot_training(summaries, hyperparam):
    train_loss, train_acc, cv_loss, cv_acc = summaries[:,0], summaries[:,1], summaries[:,2], summaries[:,3]
    fig = plt.figure(1, figsize=(12, 10))
    fig.suptitle('%s: %s'%(ENV_NAME, str(hyperparam)))

    ax = fig.add_subplot(1,2,1)
    ax.plot(train_loss, 'r-', label='Train')
    ax.plot(cv_loss, 'b-', label='CV')
    plt.title('Loss')
    plt.xlabel('# Gradient Steps')
    plt.legend()

    ax = fig.add_subplot(1,2,2)
    ax.plot(train_acc, 'r-', label='Train')
    ax.plot(cv_acc, 'b-', label='CV')
    plt.title('L2 Accuracy')
    plt.xlabel('# Gradient Steps')
    plt.legend()

weights = {'fc1': tf.get_variable('fc1', 
            shape=[NUM_OBS, FC1_SIZE],
            initializer=tf.contrib.layers.xavier_initializer()),
            'fc2': tf.get_variable('fc2',
            shape=[FC1_SIZE, FC2_SIZE],
            initializer=tf.contrib.layers.xavier_initializer()),
            'fc3': tf.get_variable('fc3',
            shape=[FC2_SIZE, FC3_SIZE],
            initializer=tf.contrib.layers.xavier_initializer()),
            'fc4': tf.get_variable('fc4',
            shape=[FC3_SIZE, NUM_ACTIONS],
            initializer=tf.contrib.layers.xavier_initializer())}

biases = {'b1': tf.get_variable("b1",
                shape=[FC1_SIZE],
                initializer=tf.constant_initializer(0.0)),
        'b2': tf.get_variable("b2",
                shape=[FC2_SIZE],
                initializer=tf.constant_initializer(0.0)),
        'b3': tf.get_variable("b3",
                shape=[FC3_SIZE],
                initializer=tf.constant_initializer(0.0)),
        'b4': tf.get_variable("b4",
                shape=[NUM_ACTIONS],
                initializer=tf.constant_initializer(0.0))}

global_step = tf.Variable(0, trainable=False)
init_learn_rate = tf.placeholder(tf.float32)
reg = tf.placeholder(tf.float32)
decay_rate = tf.placeholder(tf.float32)
learning_rate = tf.train.exponential_decay(init_learn_rate, global_step, NUM_EPOCHS_PER_DECAY*NUM_BATCHES_PER_EPOCH, decay_rate, staircase=True)
keep_prob = tf.placeholder(tf.float32)

x = tf.placeholder(tf.float32, shape=(None, NUM_OBS))
y = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))

preds = model(x, weights, biases, keep_prob)
total_loss = loss(y, preds, reg)
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

learn_rate_summary = tf.summary.scalar('learning_rate', learning_rate)

accuracy = tf.abs(tf.reduce_mean(y-preds))
learning_rates = [1e-4]
decay_rates = [.99]
l2_regs = [1e-5]
hyperparams = [[i, j, k] for i in learning_rates for j in decay_rates for k in l2_regs]
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
if len(sys.argv) > 1 and sys.argv[1] == 'train':
    x_train, x_cv, y_train, y_cv = util.load(ENV_NAME)
    for hyperparam in hyperparams:
        summaries = train(hyperparam, x_train, x_cv, y_train, y_cv, NUM_EPOCHS)
        plot_training(summaries, hyperparam)
        plt.savefig(output_dir(hyperparam)+'/train_summary.png')
        pdb.set_trace()
        saver.save(sess, output_dir(hyperparam) + '/model.ckpt')
else:
    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(init)

