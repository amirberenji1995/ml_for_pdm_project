### Setting the home directory to directories as the root, so that Damavand is importable easily

import sys
import os

project_root = os.path.abspath("../../")
if project_root not in sys.path:
    sys.path.append(project_root)

### Importing necessary libraries

import numpy as np
from dataset import HUST
import pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MinMaxScaler
from damavand.damavand.signal_processing.feature_extraction import (
    feature_extractor,
    smsa,
    rms,
    peak,
    crest_factor,
    clearance_factor,
    shape_factor,
    impulse_factor,
)
import scipy
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from hust_dataset.experiments.config import power_classifiers, Configurations
from hust_dataset.experiments.result import Result

### Instantiating from the configurations class

config = Configurations().__dict__

### Generating random seeds

random_seeds = np.random.randint(1000, size=(config["repetitions"]))


def main(config, random_seeds):

    ### Creating a dictionary to store the results

    results = Result(config)

    ### Loading the dataset

    dataset = HUST(config["dataset_base_dir"])

    ### Mining the dataset

    dataset.mine(config["dataset_mining_params"])
    df = pd.concat(dataset.data["vib"]).reset_index(drop=True)

    ### Signal/Metadata split

    signals, metadata = df.iloc[:, :-5], df.iloc[:, -5:]

    ### Encoding the faults

    fault_encoder = LabelEncoder()
    metadata["fault_encoded"] = fault_encoder.fit_transform(metadata["fault"])

    results.classes = fault_encoder.classes_

    ### Encoding the powers

    power_encoder = LabelEncoder()
    metadata["power_encoded"] = power_encoder.fit_transform(metadata["power"])

    ### Extracting time features

    time_features = {
        "mean": (np.mean, (), {}),
        "std": (np.std, (), {}),
        "smsa": (smsa, (), {}),
        "rms": (rms, (), {}),
        "peak": (peak, (), {}),
        "skew": (scipy.stats.skew, (), {}),
        "kurtosis": (scipy.stats.kurtosis, (), {}),
        "crest_factor": (crest_factor, (), {}),
        "clearance_factor": (clearance_factor, (), {}),
        "shape_factor": (shape_factor, (), {}),
        "impulse_factor": (impulse_factor, (), {}),
    }

    time_features_df = feature_extractor(signals, time_features)

    ### Feature, Target and Auxiliary declaration

    x = time_features_df
    y = metadata["fault_encoded"]
    z = metadata["power_encoded"]

    ### Looping through the random seeds

    for seed in random_seeds:
        temp_results = {
            "seed": seed,
        }

        ### Train/Test split

        x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
            x, y, z, test_size=0.25, random_state=seed
        )

        ### Binarizing the faults

        fault_binarizer = LabelBinarizer()
        y_train_bin = fault_binarizer.fit_transform(y_train)
        y_test_bin = fault_binarizer.transform(y_test)

        ### Scaling the features

        feature_scaler = MinMaxScaler()
        x_train_scaled = feature_scaler.fit_transform(x_train)
        x_test_scaled = feature_scaler.transform(x_test)

        ### Checking the experiment type

        if config["experiment_type"] == "with_additional_features":
            ### Instantiating the power classifier
            if config["power_classifier"] == "ANN":
                power_classifier = power_classifiers[config["power_classifier"]]()

            else:
                power_classifier = power_classifiers[config["power_classifier"]](
                    **config["power_classifier_params"][config["power_classifier"]]
                )

            if config["power_classifier"] == "ANN":
                power_binarizer = LabelBinarizer()
                z_train = power_binarizer.fit_transform(z_train)
                z_test = power_binarizer.transform(z_test)

                optimizer = Adam(
                    learning_rate=config["power_classifier_params"][
                        config["power_classifier"]
                    ]["lr"],
                    weight_decay=config["power_classifier_params"][
                        config["power_classifier"]
                    ]["lr"]
                    / config["power_classifier_params"][config["power_classifier"]][
                        "epochs"
                    ],
                )
                power_classifier.compile(
                    loss=config["power_classifier_params"][config["power_classifier"]][
                        "loss"
                    ],
                    optimizer=optimizer,
                    metrics=["accuracy"],
                )

            ### Training the power classifier
            power_classifier.fit(x_train_scaled, z_train)

            ### Evaluating the power classifier
            if config["power_classifier"] == "ANN":
                temp_results["power_classifier_evaluation"] = {
                    "training_accuracy": power_classifier.evaluate(
                        x_train_scaled, z_train
                    )[1],
                    "test_accuracy": power_classifier.evaluate(x_test_scaled, z_test)[
                        1
                    ],
                }

            else:
                temp_results["power_classifier_evaluation"] = {
                    "training_accuracy": power_classifier.score(
                        x_train_scaled, z_train
                    ),
                    "test_accuracy": power_classifier.score(x_test_scaled, z_test),
                }

            ### Checking the auxiliary feature type

            if config["aux_feature_type"] == "softcore":
                if config["power_classifier"] == "ANN":
                    aux_features_train = power_classifier.predict(x_train_scaled)
                    aux_features_test = power_classifier.predict(x_test_scaled)
                else:
                    aux_features_train = power_classifier.predict_proba(x_train_scaled)
                    aux_features_test = power_classifier.predict_proba(x_test_scaled)

            elif config["aux_feature_type"] == "hardcore":
                # TODO: Implement hardcore

                aux_features_train = power_classifier.predict(x_train_scaled)
                aux_features_test = power_classifier.predict(x_test_scaled)

            ### Forming the auxiliary features
            x_train_scaled = np.concatenate(
                (x_train_scaled, aux_features_train), axis=1
            )
            x_test_scaled = np.concatenate((x_test_scaled, aux_features_test), axis=1)
            print("\n\nAdditional features added.\n\n")

        ### Instantiating the fault classifier
        fault_classifer = config["fault_classifiers"][config["experiment_type"]]()

        ### Compiling the fault classifier
        opt = Adam(
            learning_rate=config["fault_classifier_training_params"]["lr"],
            weight_decay=config["fault_classifier_training_params"]["lr"]
            / config["fault_classifier_training_params"]["epochs"],
        )

        fault_classifer.compile(
            loss="categorical_crossentropy",
            optimizer=opt,
            metrics=["accuracy"],
        )

        # TODO: Add the best model recovery

        ### Training the fault classifier
        history = fault_classifer.fit(
            x=x_train_scaled,
            y=y_train_bin,
            epochs=config["fault_classifier_training_params"]["epochs"],
            validation_split=config["fault_classifier_training_params"]["val_split"],
            verbose=1,
        )

        ### Evaluating the fault classifier
        temp_results["fault_classifier_evaluation"] = {
            "training_accuracy": fault_classifer.evaluate(x_train_scaled, y_train_bin)[
                1
            ],
            "test_accuracy": fault_classifer.evaluate(x_test_scaled, y_test_bin)[1],
        }

        temp_results["fault_classifier_confusion_matrix"] = confusion_matrix(
            y_true=y_test,
            y_pred=np.argmax(fault_classifer.predict(x_test_scaled), axis=1),
        )

        results.repetitions.append(temp_results)

    return results


if __name__ == "__main__":
    ### Running the experiment
    results = main(config, random_seeds)

    ### Exporting the results
    results.export_json(base_dir="results.json")
