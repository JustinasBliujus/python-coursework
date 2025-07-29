import random

def load_and_preprocess_data(filename):
    structured_data = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()

            if '?' in line:
                continue

            data_points = line.split(',')[1:]  
            data_points = list(map(float, data_points))

            if data_points[9] == 2.0:
                data_points[9] = 0.0
            elif data_points[9] == 4.0:
                data_points[9] = 1.0

            structured_data.append(data_points)

    random.shuffle(structured_data)

    features = [row[:-1] for row in structured_data]  
    labels = [row[-1] for row in structured_data] 

    return features, labels

def split_data(features, labels, train_ratio=0.8, val_ratio=0.1):
    train_size = int(train_ratio * len(features))
    val_size = int(val_ratio * len(features))

    features_train, class_train = features[:train_size], labels[:train_size]
    features_val, class_val = features[train_size:train_size + val_size], labels[train_size:train_size + val_size]
    features_test, class_test = features[train_size + val_size:], labels[train_size + val_size:]
    
    print(f"Training set: {len(features_train)} samples")
    print(f"Validation set: {len(features_val)} samples")
    print(f"Test set: {len(features_test)} samples")
    
    return features_train, class_train, features_val, class_val, features_test, class_test

