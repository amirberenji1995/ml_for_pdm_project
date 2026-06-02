from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from hust_dataset.experiments.model_repository import (
    original_model_creator,
    with_additional_features_model_creator,
    ann_power_classifier,
)

power_classifiers = {
    "RF": RandomForestClassifier,
    "SVM": SVC,
    "ANN": ann_power_classifier,
}


class Configurations:
    def __init__(self):
        self.dataset_base_dir = "/nfs/home/amiber/ml_for_pdm_course_project/hust/HUST bearing a practical dataset for ball bearing fault diagnosis/HUST bearing dataset/"
        self.dataset_mining_params = {"win_len": 5000, "hop_len": 5000}
        self.experiment_type = "with_additional_features"
        self.power_classifier = None
        self.power_classifier_params = {
            "SVM": {"probability": True},
            "RF": {},
            "ANN": {
                "loss": "categorical_crossentropy",
                "lr": 1e-3,
                "epochs": 1500,
            },
        }
        self.aux_feature_type = "softcore"
        self.repetitions = 5
        self.test_size = 0.2
        self.fault_classifiers = {
            "original": original_model_creator,
            "with_additional_features": with_additional_features_model_creator,
        }
        self.fault_classifier_training_params = {
            "lr": 1e-3,
            "epochs": 1500,
            "val_split": 0.2,
        }
