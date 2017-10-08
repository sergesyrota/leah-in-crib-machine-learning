import os
# Needed to suppress warnings that TensorFlow was not compiled with some instructions support
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np
import argparse
import urllib.request
import time
import json
from model import cnn_model_fn

parser = argparse.ArgumentParser(description='Looks at one image to generate prediction (is Leah in crib?)')
parser.add_argument('--image', help='Path to JPEG image file or URL to download it', required=True)
parser.add_argument('--model', help='Saved model folder', required=True)
parser.add_argument('--checkpoint', help='Model checkpoint to restore from', required=False)
parser.add_argument('--image-save-path', help='If set, analyzed images will be saved in that folder', required=False)
parser.add_argument('--repeat-interval', type=int, required=False,
    help='If set, will re-check image path (or URL) and re-evaluate if changed every X seconds')
parser.add_argument('--keep-history', type=int, required=False,
    help='Only works in conjunction with repeat-interval. Will keep X data points and information about difference')
parser.add_argument('--save-history', required=False,
    help='Only works in conjunction with keep-history. Will save json-encoded history that is being tracked in a file specified.')
parser.add_argument('--print', default=False, action="store_true", required=False,
    help='Output prediction every time evaluation is ran (useful for monitoring live)')

args = parser.parse_args()

def main(args):
    # hold history of previous images and their evaluations
    history = []
    while True:
        prediction = getPrediction(path=args.image, model_dir=args.model, checkpoint=args.checkpoint,
            save_path=args.image_save_path)
        if (args.keep_history is not None):
            history.append({'time': time.time(), 'prediction': float(prediction)})
            if len(history) > args.keep_history:
                history.pop(0)
            if (args.save_history is not None):
                with open(args.save_history, 'w') as f:
                    f.write(json.dumps(history))
        if (args.print):
            print(prediction)
        if (args.repeat_interval is None):
            break
        time.sleep(args.repeat_interval)
    exit(0)

def getPrediction(path, model_dir, checkpoint=None, save_path=None):
    tf.logging.set_verbosity(tf.logging.ERROR)
    img = get_image(path, save_path=save_path)
    classifier = tf.estimator.Estimator(
          model_fn=cnn_model_fn, model_dir=model_dir)
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":img},
        shuffle=False
    )
    sess = tf.Session()
    prediction = classifier.predict(input_fn, checkpoint_path=checkpoint)
    for i, p in enumerate(prediction):
        return(p["probabilities"][1])
    #print(prediction)

def get_image(path, save_path=None):
    # if we save image, this will hold file name
    save_filename = None
    try:
        image_data = tf.constant(get_file_content(path))
    except FileNotFoundError:
        # see if it is a URL, if not a file
        response = urllib.request.urlopen(path)
        img = response.read()
        if save_path is not None:
            save_filename = save_image(img, save_path)
        image_data = tf.constant(img);
        pass
    image = tf.image.decode_jpeg(image_data, channels=1, ratio=8)
    sess = tf.Session()
    data = sess.run(image).reshape((1,160,90,1))
    sess.close()
    return np.float32(data/255)
    #return {'filename': save_filename, 'data': np.float32(data/255)}

def save_image(data, path):
    filename = '{}/{}.jpg'.format(path, time.time())
    with open(filename, 'wb') as f:
        f.write(data)
    return filename

def get_file_content(path):
    fh = open(path, 'rb')
    content = fh.read()
    fh.close()
    return content
main(args)
