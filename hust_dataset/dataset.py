import os
import scipy.io as sio
from damavand.damavand.utils import splitter
import pandas as pd


class HUST:
    def __init__(
        self,
        base_directory="hust/HUST bearing a practical dataset for ball bearing fault diagnosis/HUST bearing dataset/",
    ):
        self.base_directory = base_directory
        self.data = {"vib": []}

    def mine(self, mining_params, annotate=True):
        for file in os.listdir(self.base_directory):
            mat_contents = sio.loadmat(self.base_directory + file)
            temp_df = splitter(
                mat_contents["data"].reshape((-1)),
                win_len=mining_params["win_len"],
                hop_len=mining_params["hop_len"],
            )
            fs_value = mat_contents.get("fs", None)
            temp_df["fs"] = fs_value.item() if fs_value is not None else None
            temp_df["file"] = file

            if annotate:
                temp_df = self._annotate(temp_df)

            self.data["vib"].append(temp_df)

    def _annotate(self, df):
        df[["fault", "bearing", "power"]] = df["file"].str.extract(
            r"([A-Za-z]+)(\d)(\d+)\.mat"
        )

        return df
