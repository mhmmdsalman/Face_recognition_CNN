import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

train_data = pd.read_csv ('C:\Salman\AI\Project\celeba-dataset\list_eval_partition.csv')
print(train_data)

train_img = []
for i in tqdm(range(train_data.shape[0])):
    img = image.load_img('celeba-dataset/img_align_celeba/img_align_celeba/'+train_data['image_id'][i], target_size=(28,28,3), grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    train_img.append(img)
X = np.array(train_img)

y=train_data['partition'].values
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

model=Sequential()
model.add(Conv2D(32,(5,5), activation="relu", input_shape=(28,28,3)))
model.add(MaxPool2D(2,2))
model.add(Conv2D(64,(5,5), activation="relu"))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(3, activation="softmax"))
model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

model_path ="./model/"
os.makedirs(model_path, exist_ok=True)
model.save(model_path + 'modelled_data_CNN', overwrite=True)

est = load_model(model_path + 'modelled_data_CNN')

y_pred = est.predict(X_test)

print(np.argmax(y_pred[0]))
print(y_pred[0])
print(y_test[0])

plt.imshow(X_test[0], cmap=plt.cm.binary)
plt.show()
