from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

def get_dataset(count, image_base, max_delta = 0.04, return_original = False, desired_shape = (160*90, 1), append_axis = 1):
    # NO images
    image_path = [
        tf.constant(image_base + 'no/tmp.*'),
        tf.constant(image_base + 'yes/tmp.*'),
        tf.constant(image_base + 'no/homebot-*'),
        tf.constant(image_base + 'yes/homebot-*')
    ]
    image_files = tf.matching_files(image_path)
    filename_queue = tf.train.string_input_producer(image_files, seed=1)
    reader = tf.WholeFileReader()
    file_name, file_content = reader.read(filename_queue)
    one_image = tf.image.decode_jpeg(file_content, channels=1, ratio=8)
    #resized_image = tf.image.resize_images(one_image, [5,5])
    adjusted_image = tf.image.random_brightness(one_image, max_delta, seed=1)

    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    images = None
    labels = None
    originals = []
    for i in range(count):
        image = sess.run({
            "file_name": file_name,
            "original_image": one_image,
            "adjusted_image": adjusted_image
        })
        if '/yes/' in image['file_name'].decode('ascii'):
            image["label"] = 1
        else:
            image["label"] = 0
        if images is None:
            images = image["adjusted_image"].reshape(desired_shape)
            labels = [[image["label"]]]
        else:
            images = np.append(images, image["adjusted_image"].reshape(desired_shape), append_axis)
            labels = np.append(labels, [[image["label"]]], append_axis)
        if return_original:
            originals.append(image)
        #print(image['file_name'])
        #print("original", image["original_image"][0:5,0:5])
        #print("adjusted", image["adjusted_image"][0:5,0:5])
    coord.request_stop()
    coord.join(threads)
    return (np.float32(images/255), labels, originals)
