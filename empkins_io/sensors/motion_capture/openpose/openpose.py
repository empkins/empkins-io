import json
import os

import numpy as np
import pandas as pd

from empkins_io.utils._types import path_t

keypoint_mapping = {
    0: "Nose",
    1: "Neck",
    2: "RightShoulder",
    3: "RightElbow",
    4: "RightWrist",
    5: "LeftShoulder",
    6: "LeftElbow",
    7: "LeftWrist",
    8: "MidHip",
    9: "RightHip",
    10: "RightKnee",
    11: "RightAnkle",
    12: "LeftHip",
    13: "LeftKnee",
    14: "LeftAnkle",
    15: "RightEye",
    16: "LeftEye",
    17: "RightEar",
    18: "LeftEar",
    19: "LeftBigToe",
    20: "LeftSmallToe",
    21: "LeftHeel",
    22: "RightBigToe",
    23: "RightSmallToe",
    24: "RightHeel",
    25: "Background",
}


def get_df_from_json_dir(dir_jsons: path_t) -> pd.DataFrame:
    """Returns a DataFrame in openpose format (*) from the given directory of openpose json files.

    (*) openpose format DataFrame example:
    body_part       Nose                         Neck                      ...
    channel          pos                          pos                      ...
    axis               x        y confidence        x        y confidence  ...
    frame                                                                  ...
    0            490.773  215.688   0.871895  442.767  320.179   0.852423  ...
    ...              ...      ...        ...      ...      ...        ...  ...
    16300            NaN      NaN        NaN      NaN      NaN        NaN  ...
    16301            NaN      NaN        NaN      NaN      NaN        NaN  ...

    """
    output_all_frames_list = []
    i = 0
    # go through all openpose output files
    for file in dir_jsons.glob("*.json"):
        more_than_one_person = False
        basename = os.path.basename(file)

        # extract frame number from filename
        frame_no = int(basename.split("_")[-2])

        with open(file) as f:
            op_output = json.load(f)

        # extract keypoint data
        try:
            keypoints = np.array(op_output["people"][0]["pose_keypoints_2d"])
        except IndexError:
            # skip frame
            continue
        if len(op_output["people"]) > 1:
            # too many people in detected in frame
            keypoints[:] = np.NaN
            more_than_one_person = True

        # reformat keypoint array to shape (#keypoints, 3)
        keypoints = np.reshape(keypoints, (-1, 3))

        # eliminate values with 0 confidence
        x, y, c = np.hsplit(keypoints, 3)
        x[c == 0] = np.NaN
        y[c == 0] = np.NaN
        keypoint_dict = {"x": x.flatten(), "y": y.flatten(), "confidence": c.flatten()}

        # bring data into a useful dataframe format
        df = pd.DataFrame.from_dict(keypoint_dict, orient="columns")
        df.index.names = ["body_part"]
        df.reset_index(inplace=True)
        df["frame"] = frame_no
        df = df.pivot(index="frame", columns=["body_part"])
        df.columns = df.columns.rename(["axis", "body_part"])
        df.columns = df.columns.swaplevel(0, 1)
        df[
            "additional_info",
            "more_than_one_person",
        ] = more_than_one_person

        output_all_frames_list.append(df)

        i += 1
        if i % 1000 == 0:
            print(f"Frame {i}...")

    # put into DataFrame
    output_all_frames = pd.concat(output_all_frames_list)

    # sort both indices
    output_all_frames.sort_index(axis=1, level=[0], ascending=[True], inplace=True)
    output_all_frames.sort_index(inplace=True)

    # map keypoint numbers to body parts
    output_all_frames.rename(columns=keypoint_mapping, level=0, inplace=True)

    # add channel level
    output_all_frames = pd.concat({"pos": output_all_frames}, names=["channel"], axis=1)
    output_all_frames.columns = output_all_frames.columns.swaplevel(0, 1)

    return output_all_frames
