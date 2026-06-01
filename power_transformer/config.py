from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from model_repository import (
    original_regressor_model,
    with_features_regressor_model,
    ann_state_classifier,
)

state_classifiers = {
    "RF": RandomForestClassifier,
    "SVM": SVC,
    "ANN": ann_state_classifier,
}


class Configurations:
    def __init__(self):
        self.dataset_path = "power_transofrmer_dataset.csv"
        self.experiment_type = "with_additional_features"
        self.state_classifier = None
        self.state_classifier_params = {
            "SVM": {"probability": True},
            "RF": {},
            "ANN": {
                "loss": "categorical_crossentropy",
                "lr": 1e-3,
                "epochs": 1,
            },
        }
        self.aux_feature_type = "softcore"
        self.repetitions = 5
        self.test_size = 0.2
        self.regressor = {
            "original": original_regressor_model,
            "with_additional_features": with_features_regressor_model,
        }
        self.regressors_training_params = {
            "lr": 5e-3,
            "epochs": 1,
            "val_split": 0.2,
        }
