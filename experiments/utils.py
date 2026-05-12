import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_error_distribution(model, x_data, y_labels):
    # 1. Get the raw errors
    errors = model.get_reconstruction_errors(x_data)

    # 2. Create the plot
    plt.figure(figsize=(12, 6))

    # Get unique labels (e.g., ['Normal', 'IR_Fault', 'OR_Fault'])
    unique_labels = np.unique(y_labels)

    for label in unique_labels:
        # Mask the errors for just this class
        class_errors = errors[y_labels == label]

        # Plotting the density (KDE) for better visibility of overlaps
        sns.kdeplot(class_errors, label=f"Class: {label}", fill=True, alpha=0.4)
        # Alternatively, for a raw histogram:
        # plt.hist(class_errors, bins=50, alpha=0.5, label=f"Class: {label}")

    plt.title("Reconstruction Error Distribution by Bearing Health State")
    plt.xlabel("Mean Squared Error (Reconstruction Loss)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()
