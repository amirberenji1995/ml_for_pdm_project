import os
import h5py
from damavand.damavand.utils import splitter


class Dataset:
    def __init__(
        self,
        dir: str = "/nfs/home/amiber/ml_for_pdm_course_project/CNC_Machining/data/",
        machine: list = "M01",
        operations: list = [
            "00",
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
        ],
        channels: list = ["0", "1", "2"],
    ):

        self.dir = dir
        self.machine = machine
        self.operations = operations
        self.channels = channels
        self.data = {f"OP{op}": {c: [] for c in channels} for op in self.operations}

    def mine(self, win_len: int = 500, hop_len: int = 500):
        for op in self.operations:
            for state in os.listdir(self.dir + f"{self.machine}/" + f"OP{op}/"):
                files = os.listdir(
                    self.dir + f"{self.machine}/" + f"OP{op}/" + f"{state}/"
                )

                for file in files:
                    with h5py.File(
                        self.dir + self.machine + f"/OP{op}/" + state + "/" + file, "r"
                    ) as f:
                        print(
                            self.dir + self.machine + f"/OP{op}/" + state + "/" + file,
                            " --- ",
                            f["vibration_data"].shape,
                        )
                        for channel in self.channels:
                            temp_df = splitter(
                                f["vibration_data"][:, int(channel)], win_len, hop_len
                            )

                            temp_df["machine"] = self.machine
                            temp_df["operation"] = f"OP{op}"
                            temp_df["state"] = state
                            temp_df["file"] = file

                            self.data[f"OP{op}"][channel].append(temp_df)
        return self.data
