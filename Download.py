# Tutorial is from https://www.tensorflow.org/tutorials/images/hub_with_keras#an_imagenet_classifier

from __future__ import absolute_import, division, print_function
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

tf.VERSION

# Original Data of Flowers
data_root = tf.keras.utils.get_file(
  'flower_photos','https://drive.google.com/uc?export=download&id=174g2lMJMLNqTcgLQXnYQ9NfnrFcsWTvR',
   untar=True)
print(data_root)
