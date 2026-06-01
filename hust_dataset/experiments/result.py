import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Create a custom encoder that handles ALL numpy types automatically
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class Result:
    def __init__(self, config: dict):
        self.config = config
        self.classes = None
        self.repetitions = []

    def tabulate(self):
        seeds = [rep["seed"] for rep in self.repetitions]
        power_classification_accuracies = [
            rep["power_classifier_evaluation"]["test_accuracy"]
            for rep in self.repetitions
        ]
        fault_classification_accuracies = [
            rep["fault_classifier_evaluation"]["test_accuracy"]
            for rep in self.repetitions
        ]

        return pd.DataFrame(
            {
                "seed": seeds,
                "power_classification_accuracy": power_classification_accuracies,
                "fault_classification_accuracy": fault_classification_accuracies,
            }
        )

    def _get_serializable_config(self) -> dict:
        """Filters out non-serializable elements like function references from the config dict."""
        clean_config = {}
        for key, value in self.config.items():
            if key == "fault_classifiers":
                continue
            clean_config[key] = value
        return clean_config

    def export_json(self, base_dir: str):
        """Serializes the Result object (including config and repetitions) to a JSON file."""
        if base_dir.endswith(".json"):
            directory = os.path.dirname(base_dir) or "."
            filename = os.path.basename(base_dir)
        else:
            directory = base_dir
            filename = f"result_{self.config.get('experiment_type', 'exp')}_{self.config.get('power_classifier', 'clf')}.json"

        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)

        # We no longer need to manually mutate the confusion matrix here
        # because NumpyEncoder handles ndarrays automatically!
        export_data = {
            "config": self._get_serializable_config(),
            "repetitions": self.repetitions,
            "classes": self.classes,
        }

        # 2. Tell json.dump to use our NumpyEncoder
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=4, cls=NumpyEncoder)

        print(f"Successfully exported results to {filepath}")

    def load_json(self, filepath: str):
        """Loads data from the JSON file back into this Result instance."""
        if not filepath.endswith(".json"):
            filename = f"result_{self.config.get('experiment_type', 'exp')}_{self.config.get('power_classifier', 'clf')}.json"
            filepath = os.path.join(filepath, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved result found at {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        for key, value in data["config"].items():
            self.config[key] = value

        # Convert confusion matrices back to numpy arrays upon loading
        loaded_repetitions = []
        for rep in data["repetitions"]:
            rep_copy = rep.copy()
            if "fault_classifier_confusion_matrix" in rep_copy:
                cm_list = rep_copy["fault_classifier_confusion_matrix"]
                if cm_list is not None:
                    rep_copy["fault_classifier_confusion_matrix"] = np.array(cm_list)
            loaded_repetitions.append(rep_copy)

        self.repetitions = loaded_repetitions
        self.classes = data["classes"]
        print(f"Successfully loaded results from {filepath}")

    def confusion_matrix_visualizer(
        self, rep: int = -1, title: str = "Confusion Matrix"
    ):
        cm = self.repetitions[rep]["fault_classifier_confusion_matrix"]
        plt.figure(figsize=(6, 4), dpi=250)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.classes,
            yticklabels=self.classes,
        )
        plt.title(title)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()
