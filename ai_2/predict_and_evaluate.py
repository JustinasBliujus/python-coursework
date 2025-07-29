import math

def predict(features, weights):
    ai = weights[0]  
    for i in range(len(features)):
        ai += weights[i + 1] * features[i]  
    yi = round(1 / (1 + math.exp(-ai))) 
    return yi  

def evaluate_accuracy(features, labels, weights):
    correct_predictions = sum(1 for feature, label in zip(features, labels) if predict(feature, weights) == label)
    return correct_predictions / len(features)

def evaluate_and_print_accuracy(features, labels, weights, filename="results.txt"):
    correct_predictions = 0
    with open(filename, "w") as file:
        file.write("Expected\tPredicted\n")
        for feature, label in zip(features, labels):
            predicted = predict(feature, weights)
            file.write(f"{label}\t{predicted}\n")
            if predicted == label:
                correct_predictions += 1

    accuracy = correct_predictions / len(features)
    print(f"Accuracy: {accuracy * 100:.2f}% (Results saved in {filename})")
    return accuracy