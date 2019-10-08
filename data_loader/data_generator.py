import numpy as np


class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        self.train_files = None
        self.train_labels = None
        self.dev_labels = None
        self.dev_files = None
        self.test_files = None

        self.config.dev_len = len(self.dev_files)
        self.config.test_len = len(self.test_files)

    def next_batch(self, batch_size, step, dataset='train'):
        if dataset == 'train':
            batch_files = np.random.choice(self.train_files, self.config.batch_size, replace=True)
        elif dataset == 'dev':
            files = self.dev_files
            batch_files = files[batch_size * step:batch_size * (step + 1)]
            if len(batch_files) < self.config.batch_size:
                batch_files.extend(
                    list(np.random.choice(files, self.config.batch_size - len(batch_files), replace=False)))

        batch_labels = []
        batch_data = []

        for file in batch_files:
            batch_labels.append()
            batch_data.append()

        batch_data = np.array(batch_data)
        batch_labels = np.array(batch_labels)

        yield batch_data, batch_labels

    def next_batch_inference(self, batch_size, step):
        batch_files = self.test_files[batch_size * step:batch_size * (step + 1)]

        batch_data = []
        batch_ix = []

        for file in batch_files:
            batch_ix.append()

            batch_data.append()

        batch_data = np.array(batch_data)

        yield batch_data, batch_ix
