from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import os
import numpy as np
import tensorflow as tf
import pickle
from get_dataset import get_dataset
from tensorflow.python import debug as tf_debug
from model import cnn_model_fn
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.DEBUG)


def main():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    dataset_file = current_dir + '/images/train.npz'
    try:
        dataset = pickle.load( open( dataset_file, "rb" ) )
        train_images = dataset["train_images"]
        train_labels = dataset["train_labels"]
        test_images = dataset["test_images"]
        test_labels = dataset["test_labels"]
        original_data = dataset["original_data"]
        print("Loaded training ({}) and test ({}) sets from cache".format(train_images.shape[0], test_images.shape[0]))
    except IOError:
        #train_images, train_labels, eval_images, eval_labels = get_dataset()
        train_images, train_labels, _ = get_dataset(430*4, current_dir + '/images/train/', max_delta=0.04, desired_shape=(1,160,90,1), append_axis=0)
        print(train_images.shape, train_labels.shape)
        test_images, test_labels, original_data = get_dataset(39, current_dir + '/images/test/', max_delta=0, desired_shape=(1,160,90,1), append_axis=0, return_original=True)
        dataset = {
            "train_images": train_images,
            "train_labels": train_labels,
            "test_images": test_images,
            "test_labels": test_labels,
            "original_data": original_data
        }
        pickle.dump( dataset, open( dataset_file, "wb" ) )

    #print(original_data)
    #exit(1)

    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    print("Train labels:", train_labels.shape, train_labels)
    #exit(0)

    classifier = tf.estimator.Estimator(
          model_fn=cnn_model_fn, model_dir= current_dir + "/model_data/dev")

    #predict(classifier, test_images, test_labels, original_data)

    tensors_to_log = {"probabilities": "softmax_tensor", "true_labels": "one_hot"}
    #tensors_to_log = {"probabilities": "loss"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=24)
    #counter_hook = tf.train.StepCounterHook(every_n_steps=5)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_images},
        y=train_labels,
        batch_size=200,
        num_epochs=None,
        shuffle=True)

    merged = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(current_dir + "/model_data/dev/debug/")
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(50):
        print("i: ", i)
        classifier.train(
            input_fn = train_input_fn,
            steps=10)
            #hooks=[logging_hook])

        #print(test_images.shape, test_images)
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": test_images},
            y=test_labels,
            num_epochs=1,
            shuffle=False)
        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        print("Evaluating training set: ", eval_results)

    exit(0)

def predict(classifier, test_images, test_labels, original_data):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":test_images},
        shuffle=False
    )
    sess = tf.Session()
    prediction = classifier.predict(input_fn)
    for i, p in enumerate(prediction):
        print(original_data[i]["file_name"])
        print(p)
    #print(prediction)
    exit(0);

main()
