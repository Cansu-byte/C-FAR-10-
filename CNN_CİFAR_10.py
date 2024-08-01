import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import cifar10

# Veri setini yükle
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

input_shape = (32, 32, 3)
number_of_classes = 10

# Çıktı etiketlerini one-hot vektöre çevir
y_train = to_categorical(y_train, number_of_classes)
y_test = to_categorical(y_test, number_of_classes)

x_train, cv_x, y_train, cv_y = train_test_split(x_train, y_train, 
                                                test_size=5000, random_state=42)

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Modeli oluşturma
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(number_of_classes, activation='softmax')
    ])
    
    return model

# Modeli derleme
model = create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğitme
history = model.fit(x_train, y_train, 
                    epochs=50, 
                    batch_size=32,
                    validation_data=(cv_x, cv_y),
                    verbose=1)

# Test setinde değerlendirme
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
print("Test set accuracy: {:.4f}".format(test_accuracy))

# Eğitim sürecini görselleştirme
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
