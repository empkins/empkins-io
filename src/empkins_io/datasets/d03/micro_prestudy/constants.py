from itertools import product

__all__ = [
    "STUDY_PART_DICT_RADAR",
    "STUDY_PART_DICT_VIDEO",
    "CONDITIONS",
    "STUDY_PARTS",
    "PHASES",
    "SUBPHASES",
    "ARTIFACTS",
    "INDEX_LEVELS",
    "MOVEMENT_ARTIFACTS",
    "ARTIFACT_MAPPING",
    "VIDEO_FRAME_RATE",
]

STUDY_PART_DICT_RADAR = {
    "pre": "Pre",
    "mist": "MIST",
    "post": "Post",
}

STUDY_PART_DICT_VIDEO = {
    "pre": "pre",
    "mist": "MIST",
    "post": "post",
}

CONDITIONS = ["stress", "control"]

STUDY_PARTS = ["pre", "mist", "post"]

PHASES = {"pre": ["total", "baseline"], "mist": ["mist1", "mist2", "mist3"], "post": ["total"]}

SUBPHASES = {"pre": ["total"], "mist": ["baseline", "math", "feedback"], "post": ["total"]}

ARTIFACTS = [
    "clean",
    "all",
    "silent",
    "silent_mov",
    "silent_body",
    "silent_arm",
    "silent_head",
    "silent_occ",
    "silent_other",
    "talk",
    "talk_mov",
    "talk_body",
    "talk_arm",
    "talk_head",
    "talk_occ",
    "talk_other",
    "saliva",
    "mov",
    "body",
    "arm",
    "head",
    "occ",
    "other",
]

INDEX_LEVELS = ["subject", "condition", "study_part", "phase", "subphase", "artifact"]

MOVEMENT_ARTIFACTS = ["Upper body movement", "Head movement", "Arm movement", "Occlusion", "Other movement"]

ARTIFACT_MAPPING = {
    "silent": "Silent",
    "silent_mov": list(product(["Silent"], MOVEMENT_ARTIFACTS)),
    "silent_body": "('Silent', 'Upper body movement')",
    "silent_arm": "('Silent', 'Arm movement')",
    "silent_head": "('Silent', 'Head movement')",
    "silent_occ": "('Silent', 'Occlusion')",
    "silent_other": "('Silent', 'Other')",
    "talk": "Talking",
    "talk_mov": list(product(["Talking"], MOVEMENT_ARTIFACTS)),
    "talk_body": "('Talking', 'Upper body movement')",
    "talk_arm": "('Talking', 'Arm movement')",
    "talk_head": "('Talking', 'Head movement')",
    "talk_occ": "('Talking', 'Occlusion')",
    "talk_other": "('Talking', 'Other')",
    "saliva": "Saliva Sample",
    "mov": MOVEMENT_ARTIFACTS,
    "body": "Upper body movement",
    "arm": "Arm movement",
    "head": "Head movement",
    "occ": "Occlusion",
    "other": "Other",
}

VIDEO_FRAME_RATE = 30
