from typing import Sequence, Dict
from typing_extensions import Literal, get_args

BODY_PART = Literal[
    "Hips",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "Head",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightInHandIndex",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightInHandMiddle",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightInHandRing",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightInHandPinky",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftInHandIndex",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftInHandMiddle",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftInHandRing",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftInHandPinky",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
]

BODY_PART_GROUP = Literal["total_body", "upper_extremities", "lower_extremities", "trunk"]

BODY_PART_MAPPING: Dict[BODY_PART_GROUP, Sequence[BODY_PART]] = {
    "total_body": get_args(BODY_PART),
    "upper_extremities": [
        "RightShoulder",
        "RightHand",
        "RightArm",
        "LeftShoulder",
        "LeftArm",
        "LeftHand",
        "RightForeArm",
        "LeftForeArm",
    ],
    "lower_extremities": ["RightUpLeg", "RightLeg", "RightFoot", "LeftUpLeg", "LeftLeg", "LeftFoot"],
    "trunk": ["Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck"],
}


def get_all_body_parts() -> Sequence[BODY_PART]:
    """Return all body parts.

    Returns
    -------

    """
    return get_args(BODY_PART)


def get_body_parts_by_group(
    body_part_group: str,
) -> Sequence[BODY_PART]:
    """Return all body parts belonging to the same body part group.

    Parameters
    ----------
    body_part_group : {'total_body', 'upper_extrem', 'lower_extrem', 'body'}
        body part group.

    Returns
    -------
    list of str
        all body parts belonging to ``body_body_part_group``

    See Also
    --------
    BODY_PART_GROUP
        all available body part groups
    BODY_PARTS
        all available body parts

    """
    if body_part_group not in get_args(BODY_PART_GROUP):
        raise ValueError(
            f"Invalid 'body_part_group'! Expected one of {get_args(BODY_PART_GROUP)}, got {body_part_group}."
        )
    return BODY_PART_MAPPING[body_part_group]
