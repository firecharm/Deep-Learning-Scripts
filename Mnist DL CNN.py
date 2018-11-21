# 1) Data preparation
# 2) Build, train and evaluate the model
# 3) Report the results
import numpy as np
np.random.seed(1004)
# Data 
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True,cache=False)

# Reshape data
X = X.reshape(70000,28,28,1)

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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers

batch_size = 128
num_classes = 10
epochs = 12
input_shape = (28,28,1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation ='relu',
                 input_shape =input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()


history1 = model.fit(X_train, y_train,
                     batch_size = batch_size,
                     epochs = epochs,
                     verbose = 2,
                     validation_data=(X_test, y_test))
                     
test_loss,test_acc = model.evaluate(X_test,y_test)
print('test_acc',test_acc)
# 98.67%