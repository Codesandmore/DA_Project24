import matplotlib.pyplot as plt
from src.train_model import train_models

def plot_accuracies(nb_accuracy, dt_accuracy):
    plt.bar(['Naive Bayes', 'Decision Tree'], [nb_accuracy, dt_accuracy])
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.savefig('static/accuracy_comparison.png')  # Save the plot
    plt.close()  # Close the plot
    print("Plot saved successfully")

if __name__ == "__main__":
    # Train the models and get accuracies
    nb_accuracy, dt_accuracy = train_models()
    print(f"Naive Bayes Accuracy: {nb_accuracy}")
    print(f"Decision Tree Accuracy: {dt_accuracy}")
    
    # Now call the plot function with the accuracies
    plot_accuracies(nb_accuracy, dt_accuracy)
