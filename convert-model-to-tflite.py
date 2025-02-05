#!/usr/bin/env python3

#
# This script shows how to export the model to a TFLite format, with the custom preprocessing object.
# 

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import sys
import os

MODEL_DIR = "./results/models/"
MODEL_NAME = "kaggle-single-frame-final-model"

if len(sys.argv) > 1:
    MODEL_NAME = sys.argv[1]

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME + ".keras")

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model {MODEL_PATH} not found.")
    sys.exit(1)


class MobileNetPreprocessingLayer(Layer):
    def __init__(self, **kwargs):
        super(MobileNetPreprocessingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MobileNetPreprocessingLayer, self).build(input_shape)

    def call(self, x):
        return preprocess_input(x)

    def compute_output_shape(self, input_shape):
        return input_shape


custom_objects = {"MobileNetPreprocessingLayer": MobileNetPreprocessingLayer}


print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH, custom_objects)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops
    tf.lite.OpsSet.SELECT_TF_OPS     # enable TensorFlow ops
]

tflite_model = converter.convert()
TFLITE_PATH = os.path.join(MODEL_DIR, MODEL_NAME + ".tflite")

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Tflite model saved to: {TFLITE_PATH}")
