from base.base_train import BaseTrain
from tqdm import tqdm, trange
import numpy as np
import time, os


class ExampleTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(ExampleTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self, cur_epoch):
        num_iter = int(np.ceil(self.config.train_data_len / self.config.batch_size))
        losses = []
        accs = []
        f1s = []
        for step in range(num_iter):
            loss, acc, f1 = self.train_step(step)
            losses.append(loss)
            accs.append(acc)
            f1s.append(f1)

        loss = np.mean(losses)
        acc = np.mean(accs)
        f1 = np.mean(f1s)

        eval_loss, eval_acc, eval_f1 = self.eval()
        print(
            'Epoch: {:^3} -> loss: {:.4f}, acc: {:.4f}, f1: {:.4f} -> '
            'eval_loss: {:.4f}, eval_acc: {:.4f}, eval_f1: {:.4f}'
                .format(cur_epoch + 1, loss, acc, f1, eval_loss, eval_acc, eval_f1))

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'acc': acc,
            'f1': f1,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)

    def train_step(self, step):
        batch_x, batch_y = next(
            self.data.next_batch(self.config.batch_size, step, dataset='train'))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y}
        _, loss, acc, f1 = self.sess.run([self.model.train_step, self.model.entropy,
                                          self.model.accuracy, self.model.f1],
                                         feed_dict=feed_dict)
        return loss, acc, f1

    def eval(self):
        # print('eval dev dataset...')
        accs = []
        f1s = []
        losss = []
        for step in range(int(np.ceil(self.config.dev_len / self.config.batch_size))):
            batch_x, batch_y = next(self.data.next_batch(self.config.batch_size, step, dataset='dev'))
            feed_dict = {self.model.x: batch_x, self.model.y: batch_y}
            loss, acc, f1 = self.sess.run([self.model.entropy, self.model.accuracy, self.model.f1],
                                          feed_dict=feed_dict)
            losss.append(loss)
            accs.append(acc)
            f1s.append(f1)

        loss = np.mean(losss)
        acc = np.mean(accs)
        f1 = np.mean(f1s)
        # print('eval_acc: {:.4f}, eval_acc_fake: {:.4f}, eval_f1: {:.4f}'
        #       .format(acc, acc_fake, f1_score))

        self.model.save(self.sess, f1)

        return loss, acc, f1

    def inference(self, eval_score):
        print('inference test dataset...')

        result = {}
        for step in trange(int(np.ceil(self.config.test_len / self.config.test_batch_size))):
            batch_x1, batch_x2, batch_ix, batch_age, batch_sex = next(
                self.data.next_batch_inference(self.config.test_batch_size, step))
            feed_dict = {self.model.x1: batch_x1, self.model.x2: batch_x2, self.model.age: batch_age,
                         self.model.sex: batch_sex}
            pred = self.sess.run(self.model.logits_filter, feed_dict=feed_dict)

            for ix in range(len(batch_ix)):
                result[batch_ix[ix]] = pred[ix]

        print('inference done.')

        base_path = '/DATA/disk1/zhangming6/projects/ecg_hf/res/submit/'
        key = 'submit_{}_{}.txt'.format(eval_score, time.strftime('%m%d%H%M'))

        with open(os.path.join(base_path, key), 'w', encoding='utf-8') as fou:

            fou.write('\n')

        print('Submit write done: {}'.format(key))
