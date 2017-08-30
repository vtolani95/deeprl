import numpy as np
import tensorflow as tf
import pdb

NUM_ACTIONS = 3
NUM_OBS = 10
ENV_NAME = 'Hopper-v1'
NUM_BATCHES_PER_EPOCH = 200
NUM_EPOCHS_PER_DECAY = 10
FC1_SIZE = 10
FC2_SIZE = 5

def output_dir(params):
    return './tf/%s_%s'%(ENV_NAME, "-".join([str(param) for param in params]))

def model(input_layer, weights, biases, keep_prob):
    fc1 = tf.nn.relu(tf.matmul(input_layer, weights['fc1']+biases['b1']))
    fc2 = tf.nn.relu(tf.matmul(fc1, weights['fc2']+biases['b2']))
    dropout = tf.nn.dropout(fc2, keep_prob)
    preds = tf.matmul(dropout, weights['fc3'] + biases['b3'])
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
    num_batches = len(x_cv)/BATCH_SIZE
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

weights = {'fc1': tf.get_variable('fc1', 
            shape=[NUM_OBS, FC1_SIZE],
            initializer=tf.contrib.layers.xavier_initializer()),
            'fc2': tf.get_variable('fc2',
            shape=[FC1_SIZE, FC2_SIZE],
            initializer=tf.contrib.layers.xavier_initializer()),
            'fc3': tf.get_variable('fc3',
            shape=[FC2_SIZE, NUM_ACTIONS],
            initializer=tf.contrib.layers.xavier_initializer())}

biases = {'b1': tf.get_variable("b1",
                shape=[FC1_SIZE],
                initializer=tf.constant_initializer(0.0)),
        'b2': tf.get_variable("b2",
                shape=[FC2_SIZE],
                initializer=tf.constant_initializer(0.0)),
        'b3': tf.get_variable("b3",
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

accuracy = tf.reduce_mean(y-preds)
pdb.set_trace()
learning_rates = [1e-3]
decay_rates = [.97]
l2_regs = [.1]
hyperparams = [[i, j, k] for i in learning_rates for j in decay_rates for k in l2_regs]
#summarys = tf.summary.scalar
if sys.argv[1] == train:
    for hyperparam in hyperparams:
        train_loss, train_acc = train_epoch(x_train, y_train, hyperparam, .8, sess)
#        cv_loss, cv_acc = cross_validate(x_cv, y_cv, hyperparam, sess)
#        train_summary.value.add(tag="Train Loss", simple_value=train_loss)
#        train_summary.value.add(tag="Train Acc", simple_value=train_acc)
#        cv_summary.value.add(tag="CV Loss", simple_value=cv_loss)
#        cv_summary.value.add(tag="CV Acc", simple_value=cv_acc)
#        curr_learn_rate = sess.run(learn_rate_summary, {init_learn_rate: hyperparam[0], decay_rate: hyperparam[1]})
#        train_writer.add_summary(train_summary, j)
#        train_writer.add_summary(curr_learn_rate, j)
#        cv_writer.add_summary(cv_summary, j)
#        saver.save(sess, output_dir(hyperparam) + '/model.ckpt')
#    else:
#        init = tf.global_variables_initializer()
#        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#        sess.run(init)
#        saver.restore(sess, './model4/ckpt/model4.ckpt')
#        return model

