import random
import math
import time
from data_processing import load_and_preprocess_data, split_data
from error_analysis import compute_error, save_error_to_file
from predict_and_evaluate import evaluate_accuracy
from graphs import plot_error_vs_epochs,plot_accuracy_vs_epochs

features, labels = load_and_preprocess_data('breast-cancer-wisconsin.data')
features_train, class_train, features_val, class_val, features_test, class_test = split_data(features, labels)

def sigmoid(a):
    return 1 / (1 + math.exp(-a))

def train_model(features_train, class_train, features_val, class_val,learning_rate=0.001, epochs=2000, Emin=1e-4):
    
    w = [random.uniform(-1, 1) for _ in range(len(features_train[0]) + 1)]
    error_history_train = []
    error_history_val = []
    accuracy_history_train = []
    accuracy_history_val = []
    totalError = float('inf')
    epoch = 0
    
    start_time = time.time()
    
    while totalError > Emin and epoch < epochs:
        train_data = list(zip(features_train, class_train))
        random.shuffle(train_data) 
        xi, ti = zip(*train_data)

        totalError = 0.0  
        gradientSum = [0 for _ in range(len(features_train[0]) + 1)]  

        for features, label in zip(xi, ti):
            ai = w[0] 
            for k in range(len(features)):
                ai += w[k + 1] * features[k]

            yi = sigmoid(ai)
            error = yi - label
            totalError += error ** 2 
            
            for k in range(len(features)):
                gradientSum[k + 1] += error * features[k]

        w[0] -= learning_rate * (gradientSum[0] / len(features_train))
        for k in range(len(features_train[0])):
            w[k + 1] -= learning_rate * (gradientSum[k + 1] / len(features_train))

        epoch += 1
        error_history_train.append(totalError / len(features))
        error_history_val.append(compute_error(features_val, class_val, w, sigmoid))
        train_accuracy = evaluate_accuracy(features_train, class_train, w)
        val_accuracy = evaluate_accuracy(features_val, class_val, w)
        accuracy_history_train.append(train_accuracy)
        accuracy_history_val.append(val_accuracy)
       
        end_time = time.time()
        training_time = end_time - start_time

    save_error_to_file("train_error_history.txt", error_history_train)
    save_error_to_file("val_error_history.txt", error_history_val)
    plot_error_vs_epochs(error_history_train, error_history_val)
    plot_accuracy_vs_epochs(accuracy_history_train, accuracy_history_val)

    return w, training_time  


trained_weights, training_time  = train_model(features_train, class_train, features_val, class_val)
print(f"Training completed in {training_time:.2f} seconds.")
print("Trained weights:")
for weight in trained_weights:
    print(f"{weight}")

val_accuracy = evaluate_accuracy(features_val, class_val, trained_weights)
test_accuracy = evaluate_accuracy(features_test, class_test, trained_weights)

print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
