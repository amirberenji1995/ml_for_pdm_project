### Importing necessary libraries

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import state_classifiers, Configurations
from result import Result
from utils import duval_triangle
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument(
    "--repetitions",
    type=int,
    help="Number of repetitions for the experiment",
    default=5,
)

parser.add_argument(
    "--experiment_type",
    type=str,
    choices=["with_additional_features", "original"],
    default="original",
)

parser.add_argument(
    "--state_classifier",
    type=str,
    choices=["RF", "SVM", "ANN"],
    default=None,
)

parser.add_argument("--state_classifier_params", type=json.loads, default={})


### Instantiating from the configurations class

config = Configurations().__dict__

### Parsing the command line arguments
config["repetitions"] = parser.parse_args().repetitions
config["experiment_type"] = parser.parse_args().experiment_type
config["state_classifier"] = parser.parse_args().state_classifier
if parser.parse_args().state_classifier is not None:
    if parser.parse_args().state_classifier_params is not None:
        config["state_classifier_params"][config["state_classifier"]].update(
            parser.parse_args().state_classifier_params
        )

### Generating random seeds

random_seeds = np.random.randint(1000, size=(config["repetitions"]))


def main(config, random_seeds):

    ### Creating a dictionary to store the results

    results = Result(config)

    ### Loading the dataset

    df = pd.read_csv(config["dataset_path"])

    ### Pseduo-labeling using Duval triangle

    duval_state = []

    for index, row in df.iterrows():
        ch4_relative = (
            row["Methane"]
            / (row["Methane"] + row["Ethylene"] + row["Acethylene"])
            * 100
        )
        c2h4_relative = (
            row["Ethylene"]
            / (row["Methane"] + row["Ethylene"] + row["Acethylene"])
            * 100
        )
        c2h2_relative = (
            row["Acethylene"]
            / (row["Methane"] + row["Ethylene"] + row["Acethylene"])
            * 100
        )

        duval_state.append(duval_triangle(ch4_relative, c2h4_relative, c2h2_relative))

    df["duval_state"] = duval_state

    ### Encoding the faults

    labelencoder = LabelEncoder()
    df["duval_state_encoded"] = labelencoder.fit_transform(df["duval_state"])

    ### Feature, Target and Auxiliary declaration

    x = df.iloc[:, :14]
    y = df.iloc[:, 14:16]
    z = df.iloc[:, -1]

    ### Looping through the random seeds

    for seed in random_seeds:
        temp_results = {
            "seed": seed,
        }

        ### Train/Test split

        x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
            x, y, z, test_size=0.25, random_state=seed
        )

        ### Scaling the features

        feature_scaler = MinMaxScaler()
        x_train_scaled = feature_scaler.fit_transform(x_train)
        x_test_scaled = feature_scaler.transform(x_test)

        ### Scaling the targets

        target_scaler = MinMaxScaler()
        y_train_scaled = target_scaler.fit_transform(y_train)
        y_test_scaled = target_scaler.transform(y_test)

        ### Checking the experiment type

        if config["experiment_type"] == "with_additional_features":
            ### Instantiating the power classifier
            if config["state_classifier"] == "ANN":
                state_classifier = state_classifiers[config["state_classifier"]]()

            else:
                state_classifier = state_classifiers[config["state_classifier"]](
                    **config["state_classifier_params"][config["state_classifier"]]
                )

            if config["state_classifier"] == "ANN":
                state_binarizer = LabelBinarizer()
                z_train = state_binarizer.fit_transform(z_train)
                z_test = state_binarizer.transform(z_test)

                optimizer = Adam(
                    learning_rate=config["state_classifier_params"][
                        config["state_classifier"]
                    ]["lr"],
                    weight_decay=config["state_classifier_params"][
                        config["state_classifier"]
                    ]["lr"]
                    / config["state_classifier_params"][config["state_classifier"]][
                        "epochs"
                    ],
                )
                state_classifier.compile(
                    loss=config["state_classifier_params"][config["state_classifier"]][
                        "loss"
                    ],
                    optimizer=optimizer,
                    metrics=["accuracy"],
                )

            ### Training the state classifier
            state_classifier.fit(x_train_scaled, z_train)

            ### Evaluating the state classifier
            if config["state_classifier"] == "ANN":
                temp_results["state_classifier_evaluation"] = {
                    "training_accuracy": state_classifier.evaluate(
                        x_train_scaled, z_train
                    )[1],
                    "test_accuracy": state_classifier.evaluate(x_test_scaled, z_test)[
                        1
                    ],
                }

            else:
                temp_results["state_classifier_evaluation"] = {
                    "training_accuracy": state_classifier.score(
                        x_train_scaled, z_train
                    ),
                    "test_accuracy": state_classifier.score(x_test_scaled, z_test),
                }

            ### Checking the auxiliary feature type

            if config["aux_feature_type"] == "softcore":
                if config["state_classifier"] == "ANN":
                    aux_features_train = state_classifier.predict(x_train_scaled)
                    aux_features_test = state_classifier.predict(x_test_scaled)
                else:
                    aux_features_train = state_classifier.predict_proba(x_train_scaled)
                    aux_features_test = state_classifier.predict_proba(x_test_scaled)

            elif config["aux_feature_type"] == "hardcore":
                # TODO: Implement hardcore

                aux_features_train = state_classifier.predict(x_train_scaled)
                aux_features_test = state_classifier.predict(x_test_scaled)

            ### Forming the auxiliary features
            x_train_scaled = np.concatenate(
                (x_train_scaled, aux_features_train), axis=1
            )
            x_test_scaled = np.concatenate((x_test_scaled, aux_features_test), axis=1)
            print("\n\nAdditional features added.\n\n")

        ### Instantiating the fault classifier
        regressor = config["regressor"][config["experiment_type"]]()

        ### Compiling the fault classifier
        opt = Adam(
            learning_rate=config["regressors_training_params"]["lr"],
            weight_decay=config["regressors_training_params"]["lr"]
            / config["regressors_training_params"]["epochs"],
        )

        regressor.compile(
            loss="mse",
            optimizer=opt,
        )

        # TODO: Add the best model recovery

        ### Training the fault classifier
        history = regressor.fit(
            x=x_train_scaled,
            y=y_train_scaled,
            epochs=config["regressors_training_params"]["epochs"],
            validation_split=config["regressors_training_params"]["val_split"],
            verbose=1,
        )

        y_train_pred = regressor.predict(x_train_scaled)
        y_test_pred = regressor.predict(x_test_scaled)

        print(f"\n\n{y_train_pred.shape}, {y_train_scaled.shape}\n\n")

        ### Evaluating the fault classifier
        temp_results["regressor_evaluation"] = {
            "mse_losses": {
                "train": mean_squared_error(y_train_pred, y_train_scaled),
                "test": mean_squared_error(y_test_pred, y_test_scaled),
            },
            "mae_losses": {
                "train": mean_squared_error(y_train_pred, y_train_scaled),
                "test": mean_squared_error(y_test_pred, y_test_scaled),
            },
            "mse_health_index": {
                "train": mean_squared_error(y_train_pred[:, 0], y_train_scaled[:, 0]),
                "test": mean_squared_error(y_test_pred[:, 0], y_test_scaled[:, 0]),
            },
            "mse_life_expectation": {
                "train": mean_squared_error(y_train_pred[:, 1], y_train_scaled[:, 1]),
                "test": mean_squared_error(y_test_pred[:, 1], y_test_scaled[:, 1]),
            },
            "mae_health_index": {
                "train": mean_absolute_error(y_train_pred[:, 0], y_train_scaled[:, 0]),
                "test": mean_absolute_error(y_test_pred[:, 0], y_test_scaled[:, 0]),
            },
            "mae_life_expectation": {
                "train": mean_absolute_error(y_train_pred[:, 1], y_train_scaled[:, 1]),
                "test": mean_absolute_error(y_test_pred[:, 1], y_test_scaled[:, 1]),
            },
        }

        results.repetitions.append(temp_results)

    return results


if __name__ == "__main__":
    ### Running the experiment
    results = main(config, random_seeds)

    ### Declaring the file_name
    file_name = f"results/{config['experiment_type']}_{config['state_classifier'] if config['state_classifier'] is not None else 'None'}_{config['state_classifier_params'][config['state_classifier']] if config['state_classifier'] is not None else 'None'}.json"

    ### Exporting the results
    results.export_json(base_dir=file_name)
