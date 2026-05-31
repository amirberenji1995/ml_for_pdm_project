from sklearn.ensemble import RandomForestClassifier
from hust_dataset.experiments.model_repository import (
    original_model_creator,
    with_additional_features_model_creator,
)

power_classifiers = {
    "random_forest": RandomForestClassifier,
    # TODO: Add other power classifiers
    # "SVM": SVC,
    # "MLP": MLPClassifier,
}


class Configurations:
    def __init__(self):
        self.dataset_base_dir = "/nfs/home/amiber/ml_for_pdm_course_project/hust/HUST bearing a practical dataset for ball bearing fault diagnosis/HUST bearing dataset/"
        self.dataset_mining_params = {"win_len": 5000, "hop_len": 5000}
        self.experiment_type = "with_additional_features"
        self.power_classifier = "random_forest"
        self.power_classifier_params = {}  # Changed from None to {} to avoid keyword unpacking crashes later!
        self.aux_feature_type = "softcore"
        self.repetitions = 1
        self.test_size = 0.2
        self.fault_classifiers = {
            "original": original_model_creator,
            "with_additional_features": with_additional_features_model_creator,
        }
        self.fault_classifier_training_params = {
            "lr": 5e-3,
            "epochs": 1,
            "val_split": 0.2,
        }
