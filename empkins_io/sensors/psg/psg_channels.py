from typing import Dict, Sequence

from typing_extensions import Literal, get_args

from empkins_io.sensors.psg.psg_systems import PSG_SYSTEM

PSG_CHANNELS_SOMNO = Literal[
    "Fp1",
    "Fp2",
    "C3",
    "C4",
    "A1",
    "A2",
    "O1",
    "O2",
    "F3",
    "F4",
    "EKG II",
    "Akku",
    "Akku Stan",
    "SpO2",
    "Pulse",
    "Pleth",
    "Druck Flow",
    "RIP Abdom",
    "Mikro",
    "RIP Thora",
    "PLMr",
    "PLMl",
    "F7",
    "F8",
    "T3",
    "T4",
    "P3",
    "P4",
    "T5",
    "T6",
    "Fpz",
    "Fz",
    "Pz",
    "Oz",
    "Cz",
    "FT9",
    "FT10",
    "Fp1:A2",
    "Fp2:A1",
    "F3:A2",
    "F4:A1",
    "C3:A2",
    "C4:A1",
    "O1:A2",
    "O2:A1",
    "Summe RIP",
    "Sync."
]

PSG_CHANNELS_MESA = Literal[
    "EKG",
    "EOG-L",
    "EOG-R",
    "EMG",
    "EEG1",
    "EEG2",
    "EEG3",
    "Pres",  # Whats that?
    "Flow",
    "Snore",
    "Thor",
    "Abdo",
    "Leg",
    "Therm",
    "Pos",
    "EKG_Off",
    "EOG-L_Off",
    "EOG-R_Off",
    "EMG_Off",
    "EEG1_Off",
    "EEG2_Off",
    "EEG3_Off",
    "Pleth",
    "OxStatus",
    "SpO2",
    "HR",
    "DHR",
]

PSG_GROUP = Literal["FullPSG", "EEG", "ECG", "EOG", "SpO2", "System", "Resp", "EMG"]

PSG_GROUP_MAPPING_SOMNO: Dict[PSG_GROUP, Sequence[PSG_CHANNELS_SOMNO]] = {
    "FullPSG": get_args(PSG_CHANNELS_SOMNO),
    "EEG": [
        "Fp1",
        "Fp2",
        "C3",
        "C4",
        "A1",
        "A2",
        "O1",
        "O2",
        "F3",
        "F4",
        "F7",
        "F8",
        "T3",
        "T4",
        "P3",
        "P4",
        "T5",
        "T6",
        "Fpz",
        "Fz",
        "Pz",
        "Oz",
        "Cz",
        "FT9",
        "FT10",
        "Fp1:A2",
        "Fp2:A1",
        "F3:A2",
        "F4:A1",
        "C3:A2",
        "C4:A1",
        "O1:A2",
        "O2:A1",
    ],
    "ECG": ["EKG II", "Pulse"],
    "EOG": [],
    "SpO2": ["SpO2"],
    "System": ["Akku", "Akku Stan"],
    "Resp": ["Pleth", "Druck Flow", "RIP Abdom", "Mikro", "RIP Thora", "Summe RIP"],
    "EMG": ["PLMr", "PLMl"],
}

PSG_GROUP_MAPPING_MESA: Dict[PSG_GROUP, Sequence[PSG_CHANNELS_MESA]] = {
    "FullPSG": [get_args(PSG_CHANNELS_MESA)],
    "EEG": ["EEG1", "EEG2", "EEG3", "EEG1_Off", "EEG2_Off", "EEG3_Off",],  # Fz, Cz  # Cz, Oz  # C4, M1
    "ECG": ["EKG" "EKG_Off", "HR", "DHR",],
    "EOG": ["EOG-L", "EOG-R", "EOG-L_Off", "EOG-R_Off",],
    "SpO2": ["SpO2", "OxStatus",],
    "System": ["Therm", "Pos"],
    "Resp": ["Flow", "Snore", "Thor", "Abdo", "Pleth",],
    "EMG": ["EMG", "Leg", "EMG_Off",],  # Left & Right Leg EMG
}


def get_full_PSG(system: PSG_SYSTEM) -> Sequence[str]:
    """
    Return full PSG
    """
    if system not in get_args(PSG_SYSTEM):
        raise ValueError(f"Invalid 'system'! Expected one of {get_args(PSG_SYSTEM)}, got {system}.")
    if system == "mesa":
        return get_args(PSG_CHANNELS_MESA)
    else:
        return get_args(PSG_CHANNELS_SOMNO)


def get_psg_channels_by_group(system: str, psg_channel_group: str,) -> Sequence[str]:
    """Return all psg channels belonging to the same psg channel group.

    Parameters
    ----------
    system : {'mesa', 'somno'}
    body_part_group : {"FullPSG", "EEG","ECG", "EOG", "SpO2", "System", "Resp", "EMG"}
        psg channel group

    Returns
    -------
    list of str
        all channels belonging to ``psg_channel_group``
    """
    if system not in get_args(PSG_SYSTEM):
        raise ValueError(f"Invalid 'system'! Expected one of {get_args(PSG_SYSTEM)}, got {system}.")
    if psg_channel_group not in get_args(PSG_GROUP):
        raise ValueError(f"Invalid 'body_part_group'! Expected one of {get_args(PSG_GROUP)}, got {PSG_GROUP}.")
    if system == "mesa":
        return PSG_GROUP_MAPPING_MESA[psg_channel_group]
    else:
        return PSG_GROUP_MAPPING_SOMNO[psg_channel_group]
