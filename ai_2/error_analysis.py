def compute_error(features, labels, weights, sigmoid):
    total_error = 0.0

    for feature, label in zip(features, labels):
        ai = weights[0]  
        for i in range(len(feature)):
            ai += weights[i + 1] * feature[i]

        yi = sigmoid(ai)
        error = yi - label
        total_error += error ** 2

    return total_error 

def save_error_to_file(filename, error_history):
    with open(filename, "w") as file:
        for error in error_history:
            file.write(f"{error}\n")
