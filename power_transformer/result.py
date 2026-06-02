import json
import os
import numpy as np
import pandas as pd


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
        self.repetitions = []

    def tabulate(self):
        seeds = [rep["seed"] for rep in self.repetitions]
        state_classifier_accuracies = [
            rep["state_classifier_evaluation"]["test_accuracy"]
            if rep.get("state_classifier_evaluation")
            else None
            for rep in self.repetitions
        ]
        mse_losses = [
            rep["regressor_evaluation"]["mse_losses"]["test"]
            for rep in self.repetitions
        ]
        mae_losses = [
            rep["regressor_evaluation"]["mae_losses"]["test"]
            for rep in self.repetitions
        ]
        mse_health_index = [
            rep["regressor_evaluation"]["mse_health_index"]["test"]
            for rep in self.repetitions
        ]
        mse_life_expectation = [
            rep["regressor_evaluation"]["mse_life_expectation"]["test"]
            for rep in self.repetitions
        ]
        mae_health_index = [
            rep["regressor_evaluation"]["mae_health_index"]["test"]
            for rep in self.repetitions
        ]
        mae_life_expectation = [
            rep["regressor_evaluation"]["mae_life_expectation"]["test"]
            for rep in self.repetitions
        ]

        return pd.DataFrame(
            {
                "seed": seeds,
                "state_classifier_accuracy": state_classifier_accuracies,
                "mse_losses": mse_losses,
                "mae_losses": mae_losses,
                "mse_health_index": mse_health_index,
                "mse_life_expectation": mse_life_expectation,
                "mae_health_index": mae_health_index,
                "mae_life_expectation": mae_life_expectation,
            }
        )

    def _get_serializable_config(self) -> dict:
        """Filters out non-serializable elements like function references from the config dict."""
        clean_config = {}
        for key, value in self.config.items():
            if key == "regressor":
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
            loaded_repetitions.append(rep_copy)

        self.repetitions = loaded_repetitions
        print(f"Successfully loaded results from {filepath}")
