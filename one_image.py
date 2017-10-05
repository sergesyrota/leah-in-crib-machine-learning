import os
# Needed to suppress warnings that TensorFlow was not compiled with some instructions support
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np
import argparse
from model import cnn_model_fn

parser = argparse.ArgumentParser(description='Looks at one image to generate prediction (is leah in crib?)')
parser.add_argument('--file', help='Path to JPEG image file', required=True)
parser.add_argument('--model', help='Saved model folder', required=True)
parser.add_argument('--checkpoint', help='Model checkpoint to restore from', required=False)
args = parser.parse_args()

def main(args):
    tf.logging.set_verbosity(tf.logging.ERROR)
    img = get_image(args.file)
    classifier = tf.estimator.Estimator(
          model_fn=cnn_model_fn, model_dir=args.model)
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":img},
        shuffle=False
    )
    sess = tf.Session()
    prediction = classifier.predict(input_fn, checkpoint_path=args.checkpoint)
    for i, p in enumerate(prediction):
        print(p["probabilities"][1])
    #print(prediction)
    exit(0)

def get_image(path):
    file_content = tf.constant(get_file_content(path))
    image = tf.image.decode_jpeg(file_content, channels=1, ratio=8)
    sess = tf.Session()
    data = sess.run(image).reshape((1,160,90,1))
    sess.close()
    return np.float32(data/255)

def get_file_content(path):
    fh = open(path, 'rb')
    content = fh.read()
    fh.close()
    return content
main(args)
