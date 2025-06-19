import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


breast_cancer = sklearn.datasets.load_breast_cancer()
#print(breast_cancer)
dataframe =pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
dataframe['label'] = breast_cancer.target

#print(dataframe.shape)
#print(dataframe.isnull().sum())
#print(dataframe.describe())
print(dataframe['label'].value_counts())
X = dataframe.drop('label', axis=1)
y = dataframe['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#Standardizing the data
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

#Building the Neural Network
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow import keras

#Setting yp the layers of the Neural Network
model = keras.Sequential([
     keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(25, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid'),
])

#Compiling the model
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

#Training the model
history = model.fit(X_train_std, y_train, epochs=10,  validation_split=0.1)
#print(history)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.show()

plt.plot(history.history['loss'], label='model_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
#plt.show()

loss, accuracy = model.evaluate(X_test_std, y_test)
#print(f'Test accuracy: {accuracy:.4f}')

print(X_test.shape)
print(X_test_std[0])
predictions = model.predict(X_test_std)

print(predictions[0])


prediction_label= [np.argmax(i) for i in predictions]
print(prediction_label)


#building the predictive system
input_data = (9.504,12.44,60.34,273.9,0.1024,0.06492,0.02956,0.02076,0.1815,0.06905,0.2773,0.9768,1.909,15.7,0.009606,0.01432,0.01985,0.01421,0.02027,0.002968,10.23,15.66,65.13,314.9,0.1324,0.1148,0.08867,0.06227,0.245,0.07773)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
input_data_std = scaler.transform(input_data_reshaped)

py = model.predict(input_data_std)
py_label = np.argmax(py, axis=1)


if(py_label[0] == 0):
    print("The person is likely to have Malignant breast cancer")
else:
    print("The person is likely to have Benign breast cancer")