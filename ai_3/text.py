import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import random

data = pd.read_csv("SPD.csv")
# pasirenkami pirmi 30 dalyviu
selected_users = data['subject'].unique()[:30]
data = data[data['subject'].isin(selected_users)]
# pasirenkama pirma sesija
data = data[data['sessionIndex'] == 1]
# isimami nepozymiu stulpeliai
X = data.drop(columns=['subject', 'sessionIndex', 'rep'])

# pervadina klases is s002,s003... i 0,1,...
le = LabelEncoder()
y = le.fit_transform(data['subject'])

# 0  i (1,0,0,0,0,...,0,0), 1  i (0,1,0,0,0,...,0,0)...
y_categorical = to_categorical(y, num_classes=30)

X_array = X.values

# sumaisymas
indices = np.arange(len(X_array))
np.random.shuffle(indices)
X_array = X_array[indices]
y_categorical = y_categorical[indices]

# 8:1:1 duomenu padalijimas
total_size = len(X_array)
train_size = int(total_size * 0.8)
val_size = int(total_size * 0.1)

X_train = X_array[:train_size]
y_train = y_categorical[:train_size]

X_val = X_array[train_size:train_size + val_size]
y_val = y_categorical[train_size:train_size + val_size]

X_test = X_array[train_size + val_size:]
y_test = y_categorical[train_size + val_size:]

# auto, pozymiai (31), 1 kanalas 
X_train = X_train.reshape((-1, X_train.shape[1], 1))
X_val = X_val.reshape((-1, X_val.shape[1], 1))
X_test = X_test.reshape((-1, X_test.shape[1], 1))

# apmokymas
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.4))

model.add(Conv1D(filters=128, kernel_size=3, strides=1, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(30, activation='softmax'))

learning_rate = 0.001  
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50, batch_size=32)

# grafiku vaizdavimas
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Mokymo tikslumas', color='green')
plt.plot(history.history['val_accuracy'], label='Validavimo tikslumas', color='blue')
plt.title('Klasifikavimo tikslumas')
plt.xlabel('Epocha')
plt.ylabel('Tikslumas')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Mokymo paklaida', color='red')
plt.plot(history.history['val_loss'], label='Validavimo paklaida', color='orange')
plt.title('Klasifikavimo paklaida')
plt.xlabel('Epocha')
plt.ylabel('Paklaida')
plt.legend()
plt.tight_layout()
plt.show()

final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"Mokymo paklaida: {final_train_loss:.4f}")
print(f"Validavimo paklaida: {final_val_loss:.4f}")
print(f"Mokymo tikslumas: {final_train_acc:.4f}")
print(f"Validavimo tikslumas: {final_val_acc:.4f}")

# ivertinimas su testiniais duomenimis
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Testavimo paklaida: {test_loss:.4f}")
print(f"Testavimo tikslumas: {test_accuracy:.4f}")
# tikros ir nustatytos klases
true_classes = np.argmax(y_test, axis=1)
y_pred = model.predict(X_test)
predicted_classes = np.argmax(y_pred, axis=1)
# paima po 1 irasa is kiekvienos klases lentelei
samples_per_class = {}
for idx, label in enumerate(true_classes):
    if label not in samples_per_class:
        samples_per_class[label] = idx
    if len(samples_per_class) == 30:
        break

if len(samples_per_class) < 30:
    additional_indices = list(set(range(len(true_classes))) - set(samples_per_class.values()))
    random.shuffle(additional_indices)
    for idx in additional_indices:
        samples_per_class[true_classes[idx]] = idx
        if len(samples_per_class) == 30:
            break

print(f"{'Indeksas':<8} {'Tikr. klase':<12} {'Prognoze':<10}")
print("-" * 32)
for i, idx in enumerate(samples_per_class.values()):
    true_class = true_classes[idx]
    predicted_class = predicted_classes[idx]
    print(f"{idx:<8} {true_class:<12} {predicted_class:<10}")