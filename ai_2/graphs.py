import matplotlib.pyplot as plt

def plot_error_vs_epochs(error_history_train, error_history_val):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(error_history_train)), error_history_train, label="Training Error", color='blue')
    plt.plot(range(len(error_history_val)), error_history_val, label="Validation Error", color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("Error vs. Epochs")
    plt.legend()
    plt.grid()
    plt.savefig("error_vs_epochs.png")  
    plt.show()

def plot_accuracy_vs_epochs(accuracy_history_train, accuracy_history_val):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(accuracy_history_train)), accuracy_history_train, label="Training Accuracy", color='green')
    plt.plot(range(len(accuracy_history_val)), accuracy_history_val, label="Validation Accuracy", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epochs")
    plt.legend()
    plt.grid()
    plt.savefig("accuracy_vs_epochs.png") 
    plt.show()
