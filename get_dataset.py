from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

def get_dataset(sess):
    image_base='/Users/sergeysyrota/git-source/tensorflow/leah-in-crib/images/'

    # NO images
    no_image_path=tf.constant(image_base + 'no/tmp.*')
    no_image_files = tf.matching_files(no_image_path)
    no_filename_queue = tf.train.string_input_producer(no_image_files)
    reader = tf.WholeFileReader()
    no_key, no_value = reader.read(no_filename_queue)
    no_one_image = tf.to_float(tf.image.decode_jpeg(no_value, channels=1, ratio=8))
    # YES images
    yes_image_path=tf.constant(image_base + 'yes/tmp.*')
    yes_image_files = tf.matching_files(yes_image_path)
    yes_filename_queue = tf.train.string_input_producer(yes_image_files)
    reader = tf.WholeFileReader()
    yes_key, yes_value = reader.read(yes_filename_queue)
    yes_one_image = tf.to_float(tf.image.decode_jpeg(yes_value, channels=1, ratio=8))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    train_images = None
    train_labels = None
    eval_images = None
    eval_labels = None
    np.random.seed(8579)  # Fixed seed, so the same images fall into train and eval sets on consecutive runs
    # no images
    for i in range(30):
        flat_image = sess.run(no_one_image).reshape((1, 90, 160, 1))
        if (train_images is None):
            train_images = flat_image
            train_labels = [0]
        elif (eval_images is None):
            eval_images = flat_image
            eval_labels = [1]
        # randomly distribute between train and eval sets, but not really, as we are using a constant seed
        elif (np.random.randint(0, 10) < 1):
            eval_images = np.append(eval_images, flat_image, 0)
            eval_labels = np.append(eval_labels, [0], 0)
        else:
            train_images = np.append(train_images, flat_image, 0)
            train_labels = np.append(train_labels, [0], 0)
        #print(sess.run(one_image).reshape((90*160,1)).shape)
    # yes images
    for i in range(30):
        flat_image = sess.run(yes_one_image).reshape((1,90,160,1))
        if (np.random.randint(0,10) < 1):
            eval_images = np.append(eval_images, flat_image, 0)
            eval_labels = np.append(eval_labels, [1], 0)
        else:
            train_images = np.append(train_images, flat_image, 0)
            train_labels = np.append(train_labels, [1], 0)
        #print(sess.run(one_image).reshape((90*160,1)).shape)
    #print(train_images, train_labels)
    #print(eval_images, eval_labels)
    print(train_labels.shape)
    print(eval_labels.shape)
    coord.request_stop()
    coord.join(threads)
    return (train_images, train_labels, eval_images, eval_labels)
