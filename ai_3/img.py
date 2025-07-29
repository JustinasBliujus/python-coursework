import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.activations import swish
import numpy as np

data = tf.keras.utils.image_dataset_from_directory("data", batch_size=32) #duomenys cia sumaisomi ir paimami

data = data.map(lambda x, y: (x / 255, y)) #normalizavimas padalinant iš 255
iterator = data.as_numpy_iterator()
batch = iterator.next() 

#duomenu padalijimas
# 40% train, 5% val, 5% test, tai santykis 8:1:1
train_size = int(len(data) * 0.4)
val_size = int(len(data) * 0.05)
test_size = int(len(data) * 0.05)
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)
#print(len(train), len(val), len(test))

#apmokymas
model = Sequential() #filtrai,kernel,zingsnis         256x256x3rgb
model.add(Conv2D(8, (2,2), 2, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(16, (2,2), 2, activation='relu'))
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
#loss = binary_crossentropy arba hinge
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

logdir="logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
history = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])

#grafikai
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# Paklaidos grafikas
axs[0].plot(history.history['loss'], label='Mokymo paklaida', color='teal')
axs[0].plot(history.history['val_loss'], label='Validacijos paklaida', color='orange')
axs[0].set_title('Paklaida')
axs[0].set_xlabel('Epocha')
axs[0].set_ylabel('Paklaida')
axs[0].legend()
# Tikslumo grafikas
axs[1].plot(history.history['accuracy'], label='Mokymo tikslumas', color='green')
axs[1].plot(history.history['val_accuracy'], label='Validacijos tikslumas', color='blue')
axs[1].set_title('Tikslumas')
axs[1].set_xlabel('Epocha')
axs[1].set_ylabel('Tikslumas')
axs[1].legend()
plt.tight_layout()
plt.show()
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"Mokymo paklaida: {final_train_loss:.4f}")
print(f"Validacijos paklaida: {final_val_loss:.4f}")
print(f"Mokymo tikslumas: {final_train_acc:.4f}")
print(f"Validacijos tikslumas: {final_val_acc:.4f}")

#modelio ivertinimas su testiniais duomenimis
acc = BinaryAccuracy()
y_true = []
y_pred = []
last_x_batch = None  

for batch in test.as_numpy_iterator():
    x, y = batch
    last_x_batch = x  
    predicted = model.predict(x)
    preds = np.round(predicted).astype(int)
    y_true.extend(y)
    y_pred.extend(preds)
#30 vaizdu klasifikavimo vizualizavimas
last_images = last_x_batch[-30:]
last_true_labels = y_true[-30:]
last_pred_labels = y_pred[-30:]
plt.figure(figsize=(15, 15))
for i in range(30):
    plt.subplot(6, 5, i + 1)
    plt.imshow(last_images[i].astype("float32"))
    plt.title(f"Actual: {int(last_true_labels[i])}\nPred: {last_pred_labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()