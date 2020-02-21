#!/usr/bin/env python3

# Modified from https://raw.githubusercontent.com/bioinf-jku/TTUR/master/precalc_stats_example.py

import os
import glob
import numpy as np
import fid
from imageio import imread
import tensorflow as tf
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_path", type=str, help='path to training set images')
parser.add_argument("--output_path", type=str, help='path for where to store the statistics')
parser.add_argument("-i", "--inception", type=str, default=None, help='Path to Inception model (will be downloaded if not provided)')

args = parser.parse_args()

inception_path = fid.check_or_download_inception(args.inception) # download inception if necessary
data_path = args.data_path
output_path = args.output_path
print("ok")
# loads all images into memory (this might require a lot of RAM!)
print("load images..", end=" " , flush=True)
image_list = glob.glob(os.path.join(data_path, '*.jpg'))
images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])
print("%d images found and loaded" % len(images))

print("create inception graph..", end=" ", flush=True)
fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
print("ok")

print("calculte FID stats..", end=" ", flush=True)
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
    np.savez_compressed(output_path+'/fid_stats.npz', mu=mu, sigma=sigma)
print("finished")
