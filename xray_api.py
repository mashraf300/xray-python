from flask import Flask
from flask_restful import Resource, Api, reqparse
import ast
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)
    
print(tf.__version__)

IMAGE_SIZE = [180, 180]
BATCH_SIZE = 16 * strategy.num_replicas_in_sync

def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == "PNEUMONIA"

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, IMAGE_SIZE)

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label



app = Flask(__name__)
api = Api(app)


class XRay(Resource):
    def get(self):
        data = "1"
        test_list_ds = tf.data.Dataset.list_files(str('test/*/*'))
        TEST_IMAGE_COUNT = tf.data.experimental.cardinality(test_list_ds).numpy()

        print(TEST_IMAGE_COUNT)


        test_ds = test_list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.batch(BATCH_SIZE)


        reconstructed_model = keras.models.load_model("xray_model.h5")

        y = reconstructed_model.predict_classes(test_ds)

        print(y[0][0])
        return {'output': int(y[0][0])}, 200

    pass

api.add_resource(XRay, '/xray')

if __name__ == '__main__':
    app.run(Debug=False, host='0.0.0.0')