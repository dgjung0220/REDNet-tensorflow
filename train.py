# Donggoo Jung (dgjung0220@gmail.com)
# https://dgjung.me

from argparse import ArgumentParser

from utils import get_hr_and_lr
from utils import psnr_metric

from model import REDNet

import tensorflow as tf
import os
import pathlib
import datetime

_available_optimizers = {
    "adam": tf.optimizers.Adam,
}

def main(args):

    image_root = pathlib.Path(os.getcwd() + '/' + args.dataset)
    all_image_paths = list(image_root.glob('*/*'))

    train_path, valid_path, test_path = [], [], []

    for image_path in all_image_paths:

        if str(image_path).split('.')[-1] != 'jpg':
            continue

        type = str(image_path).split('\\')[-2]

        if type == 'train':
            train_path.append(str(image_path))
        elif type == 'val':
            valid_path.append(str(image_path))
        else:
            test_path.append(str(image_path))

    train_dataset = tf.data.Dataset.list_files(train_path)
    train_dataset = train_dataset.map(get_hr_and_lr)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(args.train_batch_size)

    valid_dataset = tf.data.Dataset.list_files(valid_path)
    valid_dataset = valid_dataset.map(get_hr_and_lr)
    valid_dataset = valid_dataset.repeat()
    valid_dataset = valid_dataset.batch(args.valid_batch_size)

    model = REDNet(args.num_layers)
    model.compile(optimizer=args.optimizer(args.lr), loss='mse', metrics=[psnr_metric])

    log_dir = os.getcwd() + "\logs\\" + args.logdir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.summary()

    history = model.fit(train_dataset,
                        epochs=args.num_epoch,
                        steps_per_epoch=len(train_path) // args.train_batch_size,
                        validation_data=valid_dataset,
                        validation_steps=len(valid_path),
                        verbose=2,
                        callbacks=[tensorboard_callback])

    model.save_weights(f"mobilenetv3_{args.model_type}_{args.dataset}_{args.num_epoch}.h5")

    layers = args.num_layers * 2
    model.save_weights(f"REDNet-{layers}.h5")

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--num_layers", type=int, default=15)   # REDNet-30
    parser.add_argument("--dataset", type=str, default='dataset/bsd_images')
    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--valid_batch_size", type=int, default=1)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="adam", choices=_available_optimizers.keys())

    parser.add_argument("--logdir", type=str, default='logs')


    args = parser.parse_args()
    main(args)