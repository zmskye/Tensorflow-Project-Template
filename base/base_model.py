import tensorflow as tf
import os
import re


class BaseModel:
    def __init__(self, config):
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()

        self.best_score = 0

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess, score):
        # print("Saving model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        if score > self.best_score:
            self.saver_best.save(sess, os.path.join(self.config.ckpt_best, 'score_{:.4f}'.format(score)),
                                 self.global_step_tensor)
            self.best_score = score
        # print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        if self.config.use_best:
            latest_checkpoint = tf.train.latest_checkpoint(self.config.ckpt_best)
        else:
            latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        # latest_checkpoint = self.config.ckpt_best + 'score_0.7097-6600'
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            eval_score = re.findall(r'score_(.*)-', latest_checkpoint)
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")
            if eval_score:
                return eval_score[0]
        return 0

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError
