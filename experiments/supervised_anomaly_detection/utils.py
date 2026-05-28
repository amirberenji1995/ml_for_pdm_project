import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
from sklearn.metrics import classification_report, confusion_matrix


class VibrationClassificationDataset(Dataset):
    def __init__(self, x_data, y_data):
        # Ensure clean NumPy conversions to avoid pandas indexing errors
        self.x = torch.tensor(np.asarray(x_data), dtype=torch.float32).unsqueeze(
            1
        )  # Shape: (N, 1, 1000)

        if hasattr(y_data, "to_numpy"):
            self.y = torch.tensor(y_data.to_numpy(), dtype=torch.long)
        else:
            self.y = torch.tensor(np.asarray(y_data), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def create_balanced_contrastive_pairs(x_train, y_train):
    """
    Downsamples the majority class and generates all possible pairs.
    Handles y_train whether it is a NumPy array or a pandas Series.
    """
    # Convert y_train to a raw numpy array to bypass pandas label-index matching
    if hasattr(y_train, "to_numpy"):
        y_train_arr = y_train.to_numpy()
    else:
        y_train_arr = np.asarray(y_train)

    # Ensure x_train is also a clean numpy array for safe indexing
    x_train_arr = np.asarray(x_train)

    # Identify position-based indices
    good_indices = np.where(y_train_arr == 0)[0]
    bad_indices = np.where(y_train_arr == 1)[0]

    # Downsample good indices to match bad indices exactly (68 samples)
    np.random.seed(42)
    downsampled_good_indices = np.random.choice(
        good_indices, size=len(bad_indices), replace=False
    )

    # Combine into a balanced pool (136 samples total)
    balanced_indices = np.concatenate([downsampled_good_indices, bad_indices])

    # Now positional indexing works flawlessly on both arrays
    x_balanced = x_train_arr[balanced_indices]
    y_balanced = y_train_arr[balanced_indices]

    # Generate all unique pairs (index combinations)
    num_samples = len(y_balanced)
    pair_indices = list(combinations(range(num_samples), 2))

    pairs_x1 = []
    pairs_x2 = []
    pair_labels = []

    for idx1, idx2 in pair_indices:
        pairs_x1.append(x_balanced[idx1])
        pairs_x2.append(x_balanced[idx2])

        # Contrastive target: 1 if identical states, 0 if distinct states
        label = 1.0 if y_balanced[idx1] == y_balanced[idx2] else 0.0
        pair_labels.append(label)

    # Convert arrays to tensors
    X1 = torch.tensor(np.array(pairs_x1), dtype=torch.float32).unsqueeze(1)
    X2 = torch.tensor(np.array(pairs_x2), dtype=torch.float32).unsqueeze(1)
    Y_pairs = torch.tensor(np.array(pair_labels), dtype=torch.float32)

    return X1, X2, Y_pairs


# PyTorch Dataset wrapper for pairs
class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, x1, x2, y):
        self.x1 = x1
        self.x2 = x2
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx]


def evaluate_model(
    model,
    x,
    y,
    target_names=["good", "bad"],
    config: dict = None,
    export_path: str = None,
):
    """
    Evaluates the model classification performance on given features (x) and labels (y).
    Optionally logs the exact console output to a text file.

    Parameters:
        model: The PyTorch model to evaluate.
        x: Features (Pandas DataFrame or NumPy array).
        y: Target labels (Pandas Series or NumPy array).
        target_names: List of strings for the classification report classes.
        config: Experiment configuration dictionary.
        export_path: Path to the .txt file where results should be appended/written.
    """
    model.set_mode("classify")
    model.eval()

    # Create a temporary dataset and loader for clean batch processing
    eval_dataset = VibrationClassificationDataset(x, y)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in eval_loader:
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            _, predicted = torch.max(probabilities, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # --- Generate the text content safely ---
    report_text = classification_report(
        all_labels, all_preds, target_names=target_names
    )
    cm = confusion_matrix(all_labels, all_preds)

    # Format config string nicely line-by-line
    config_str = ""
    if config:
        config_str = "\n".join([f"  {k}: {v}" for k, v in config.items()])
    else:
        config_str = "  None"

    # Assemble everything into a single, clean multi-line string
    output_lines = [
        "\n============= Experiment configuration =============",
        "config:",
        config_str,
        "\n================ Evaluation Metrics ================",
        report_text,
        "Confusion Matrix:",
        f"True {target_names[0].capitalize()}: {cm[0, 0]} | False {target_names[1].capitalize()}: {cm[0, 1]}",
        f"False {target_names[0].capitalize()}: {cm[1, 0]} | True {target_names[1].capitalize()}: {cm[1, 1]}\n",
    ]
    full_output = "\n".join(output_lines)

    # 1. Print to the console as usual
    print(full_output)

    # 2. Write to the text file if a path is provided
    if export_path:
        # Using "a" (append) mode so you don't overwrite previous runs,
        # change to "w" if you always want a fresh file every single time.
        with open(export_path, "a") as f:
            f.write(full_output)
            f.write("\n" + "=" * 50 + "\n")  # Visual separator between runs
        print(f"--> Performance report successfully saved to: {export_path}")

    return {
        "predictions": all_preds,
        "true_labels": all_labels,
        "classification_report": report_text,
        "confusion_matrix": cm,
    }


def contrastive_loss(z1, z2, label, margin=1.0):
    dist = torch.nn.functional.pairwise_distance(z1, z2)
    loss_pos = label * torch.pow(dist, 2)
    loss_neg = (1 - label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    return torch.mean(loss_pos + loss_neg)
