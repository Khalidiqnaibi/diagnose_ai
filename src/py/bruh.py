import tensorflow as tf
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] =  "0"
print(tf.__version__)
