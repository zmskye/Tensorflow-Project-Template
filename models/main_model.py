import tensorflow as tf
import tensorflow.contrib as tf_contrib
from base.base_model import BaseModel
import os, sys
import numpy as np

cur_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(cur_dir, '../../'))

TRAINING = tf.Variable(initial_value=True, dtype=tf.bool, trainable=False)

from models.template_model import TmModel
from utils.learning_schedules import cosine_decay_with_warmup


class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)
        print('init_learning_rate: {}, batch size: {}, ckpt_dir: {}'
              .format(self.config.learning_rate, self.config.batch_size, self.config.exp_name))

        self.label_num = 55

        self.steps_every_epoch = int(self.config.train_data_len / self.config.batch_size)

        self.learning_rate_compute()

        self.build_model()
        self.init_saver()

    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.
        self.x = tf.placeholder(tf.float32, shape=[None, 7680, 12])

        self.y = tf.placeholder(tf.float32, shape=[None, self.label_num])

        self.logits = TmModel().build_model(self.x, num_class=self.label_num)

        with tf.name_scope('loss'):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.logits)
            # loss = tf.multiply(loss, tf.cast(self.weights, tf.float32))
            self.entropy = tf.reduce_mean(loss)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.learning_rate_decay) \
                    .minimize(self.entropy, global_step=self.global_step_tensor)

                # correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
                # self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                self.metric_compute(self.logits)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        self.saver_best = tf.train.Saver(max_to_keep=10)

    def metric_compute(self, logits):
        # self.logits_filter = tf.where(logits > 0.5, tf.ones_like(logits), tf.zeros_like(logits))
        self.logits_filter = tf.round(tf.sigmoid(logits))
        correct = tf.equal(self.logits_filter, self.y)
        correct_sum_a = tf.reduce_sum(tf.reduce_min(tf.cast(correct, tf.float32), 1))
        self.accuracy = correct_sum_a / self.config.batch_size

        correct_num = tf.reduce_sum(tf.boolean_mask(self.logits_filter, self.y))
        P = correct_num / tf.reduce_sum(self.logits_filter)
        R = correct_num / tf.reduce_sum(self.y)
        self.f1 = (2 * P * R) / (P + R)

    def learning_rate_compute(self):
        if not self.config.cosine_decay_warmup:
            print(f'exponential decay from {self.config.learning_rate}')
            self.learning_rate_decay = tf.train.exponential_decay(self.config.learning_rate, self.global_step_tensor,
                                                                  self.steps_every_epoch, self.config.decay_rate,
                                                                  staircase=True)
        else:
            print(f'cosine_decay_with_warmup: {1e-6}')
            total_steps = self.config.num_epochs * self.steps_every_epoch
            self.learning_rate_decay = cosine_decay_with_warmup(self.global_step_tensor, self.config.learning_rate,
                                                                total_steps, warmup_learning_rate=1e-6,
                                                                warmup_steps=self.steps_every_epoch,
                                                                hold_base_rate_steps=self.steps_every_epoch)
