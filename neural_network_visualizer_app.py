# -*- coding: utf-8 -*-
"""Neural network visualizer app

# Import libraries
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""# Download Data"""

(x_train, y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()

"""# Plot Examples"""

plt.figure(figsize=(10,10))

for i in range (0,16):
  plt.subplot(4,4,i+1)
  plt.imshow(x_train[i],cmap='binary')
  plt.xlabel(str(y_train[i]))
  plt.xticks([])
  plt.yticks([])
plt.show()

"""#Normalize Data"""

x_train = np.reshape(x_train,(60000, 28*28))
x_test = np.reshape(x_test,(10000, 28*28))

x_train = x_train/255
x_test = x_test/255

"""# Create Neural Network"""

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation= 'sigmoid', input_shape=(784,)),
    tf.keras.layers.Dense(32, activation = 'sigmoid'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

model.compile(
    loss= 'sparse_categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)

"""# Training the model"""

_ = model.fit( x_train,y_train,
              validation_data = (x_test,y_test),
              epochs = 20,batch_size = 1048,
              verbose = 2
)

"""# Save Model"""

model.save('model.h5')

"""# ML Server"""

# Commented out IPython magic to ensure Python compatibility.
# %%writefile ml_server.py
# import json
# import tensorflow as tf
# import numpy as np
# import random
# from flask import Flask, request
# 
# app = Flask(__name__)
# 
# model = tf.keras.models.load_model('model.h5')
# feature_model = tf.keras.models.Model(
#     model.imputs,
#     [layer.output for layer in model.layers]
# )
# 
# #download dataset (test examples)
# _, (x_test,_) = tf.keras.datasets.mnist.load_data()
# x_test = x_test/255
# 
# def get_perdiction():
#   index = np.random.choice(x_test.shape[0])
#   image = x_test[index,:,:]
#   image_arr = np.reshape(image,(1,784))
#   retur feature_model.perdict(image_arr), image
# 
# @app.route('/', methods = ['GET' , 'POST'])
# def index():
#   if request.method == 'POST':
#     preds, image = get_prediction()
#     final_preds = [p.tolist() for p in preds]
#     return json.dumps({
#         'prediction' : final_preds,
#         'image' : image.tolist()
#     })
#   return 'Welcome to the model server!'
# 
# if __name__ == .'__main__':
#   app.run()

"""# Streamlit web app"""

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# 
# import streamlit as st
# import json
# import requests
# import matplotlib.pyplot as plt
# import numpy as np
# 
# URL = 'http://127.0.01:5000'
# 
# st.title('Neural Network Visualizer')
# st.sidebar.markdown('##Input Image')
# 
# if st.button('Get random prediction'):
#   response = request.post(URL, data={} )
#   response = json.load(response.text)
#   preds = response.get('prediction')
#   image = response.get('image')
#   image = np.reshape(image, (28,28))
# 
#   st.sidebar.image(image, width = 150) 
# 
#   for layer, p in enumerate(preds):
#             numbers = np.squeeze(np.array(p))
#     
#             plt.figure(figsize=(32, 4))
#     
#             if layer == 2:
#                 row = 1
#                 col = 10
#             else:
#                 row = 2
#                 col = 16
#     
#             for i, number in enumerate(numbers):
#                 plt.subplot(row, col, i + 1)
#                 plt.imshow((number * np.ones((8, 8, 3))).astype('float32'), cmap='binary')
#                 plt.xticks([])
#                 plt.yticks([])
#                 if layer == 2:
#                     plt.xlabel(str(i), fontsize=40)
#             plt.subplots_adjust(wspace=0.05, hspace=0.05)
#             plt.tight_layout()
#     
#             st.text('Layer {}'.format(layer + 1), )
#             st.pyplot()

