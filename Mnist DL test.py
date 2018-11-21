# 1) Data preparation
# 2) Build, train and evaluate the model
# 3) Report the results
import numpy as np
np.random.seed(1004)
# Data 
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True,cache=False)

# Data Exploration
# import pandas as pd
# df =  pd.DataFrame(X)
# df['label'] = y
# df.head(5)

# Reshape data
# X.reshape(70000,28,28)
# Necessary?

# Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Since X are numbers out of 255, in type float 64, conversion is needed. Convert to out of 1
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# convert labels to OHE
from keras import utils
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

# Build NN Structure
from keras.layers import *
from keras import Model
from keras import optimizers
n_input = 784 # number of features
n_hidden_1 = 300
n_hidden_2 = 100
n_hidden_3 = 100
n_hidden_4 = 200
num_digits = 10

Inp = Input(shape=(784,))
x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)
x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)
x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)
x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)
output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)

model = Model(Inp,output)
# model.summary()

# Setting the hyper parameters
learning_rate=0.1
training_epochs = 20
batch_size = 100
adam = optimizers.Adam(lr=learning_rate)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history1 = model.fit(X_train, y_train,
                     batch_size = batch_size,
                     epochs = training_epochs,
                     verbose = 2,
                     validation_data=(X_test, y_test))
                     
test_loss,test_acc = model.evaluate(X_test,y_test)
print('test_acc',test_acc)