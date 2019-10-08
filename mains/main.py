import tensorflow as tf

import os, sys

cur_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(cur_dir, '../'))

from data_loader.data_generator import DataGenerator
from models.example_model import ExampleModel
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        # args = get_args()
        config_file = os.path.join(cur_dir, '../configs/example.json')
        config = process_config(config_file)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir, config.ckpt_best])

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    # create tensorflow session
    sess = tf.Session(config=tf_config)
    # create your data generator
    data = DataGenerator(config)

    # create an instance of the model you want
    model = ExampleModel(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = ExampleTrainer(sess, model, data, data.config, logger)
    # load model if exists
    eval_score = model.load(sess)
    # here you train your model
    trainer.train()

    # trainer.inference(eval_score)


if __name__ == '__main__':
    main()
