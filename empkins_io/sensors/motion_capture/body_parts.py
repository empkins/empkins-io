from typing import Sequence, Dict

from typing_extensions import Literal, get_args

SYSTEMS = Literal["perception_neuron", "xsens"]

BODY_PART_PERCEPTION_NEURON = Literal[
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

BODY_PART_XSENS = Literal[
    "Head",
    "L3",
    "L5",
    "LeftFoot",
    "LeftForeArm",
    "LeftHand",
    "LeftLowerLeg",
    "LeftShoulder",
    "LeftToe",
    "LeftUpperArm",
    "LeftUpperLeg",
    "Neck",
    "Pelvis",
    "RightFoot",
    "RightForeArm",
    "RightHand",
    "RightLowerLeg",
    "RightShoulder",
    "RightToe",
    "RightUpperArm",
    "RightUpperLeg",
    "T12",
    "T8",
]

BODY_PART_GROUP = Literal["TotalBody", "UpperExtremities", "LowerExtremities", "Trunk"]

BODY_PART_MAPPING_PERCEPTION_NEURON: Dict[
    BODY_PART_GROUP, Sequence[BODY_PART_PERCEPTION_NEURON]
] = {
    "TotalBody": get_args(BODY_PART_PERCEPTION_NEURON),
    "UpperExtremities": [
        "RightShoulder",
        "RightHand",
        "RightArm",
        "LeftShoulder",
        "LeftArm",
        "LeftHand",
        "RightForeArm",
        "LeftForeArm",
    ],
    "LowerExtremities": [
        "RightUpLeg",
        "RightLeg",
        "RightFoot",
        "LeftUpLeg",
        "LeftLeg",
        "LeftFoot",
    ],
    "Trunk": ["Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck"],
}

BODY_PART_MAPPING_XSENS: Dict[BODY_PART_GROUP, Sequence[BODY_PART_XSENS]] = {
    "TotalBody": get_args(BODY_PART_XSENS),
    "UpperExtremities": [
        "RightShoulder",
        "RightUpperArm",
        "RightForeArm",
        "RightHand",
        "LeftShoulder",
        "LeftUpperArm",
        "LeftForeArm",
        "LeftHand",
    ],
    "LowerExtremities": [
        "RightLowerLeg",
        "RightFoot",
        "RightToe",
        "LeftLowerLeg",
        "LeftFoot",
        "LeftToe",
    ],
    "Trunk": [
        "L3",
        "L5",
        "T8",
        "T12",
        "Neck",
    ],
}


def get_all_body_parts(system: str) -> Sequence[str]:
    """Return all body parts.

    Parameters
    ----------
    system : {'perception_neuron', 'xsens'}

    Returns
    -------

    """
    if system not in get_args(SYSTEMS):
        raise ValueError(
            f"Invalid 'system'! Expected one of {get_args(SYSTEMS)}, got {system}."
        )
    if system == "perception_neuron":
        return get_args(BODY_PART_PERCEPTION_NEURON)
    else:
        return get_args(BODY_PART_XSENS)


def get_body_parts_by_group(
    system: str,
    body_part_group: str,
) -> Sequence[str]:
    """Return all body parts belonging to the same body part group.

    Parameters
    ----------
    system : {'perception_neuron', 'xsens'}
    body_part_group : {'TotalBody', 'upper_extrem', 'lower_extrem', 'body'}
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
    if system not in get_args(SYSTEMS):
        raise ValueError(
            f"Invalid 'system'! Expected one of {get_args(SYSTEMS)}, got {system}."
        )
    if body_part_group not in get_args(BODY_PART_GROUP):
        raise ValueError(
            f"Invalid 'body_part_group'! Expected one of {get_args(BODY_PART_GROUP)}, got {body_part_group}."
        )
    if system == "perception_neuron":
        return BODY_PART_MAPPING_PERCEPTION_NEURON[body_part_group]
    else:
        return BODY_PART_MAPPING_XSENS[body_part_group]
