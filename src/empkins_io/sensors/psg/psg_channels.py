from typing import Dict, Literal, Sequence

from typing_extensions import get_args

from empkins_io.sensors.psg.psg_systems import PSG_SYSTEM

PSG_CHANNELS_SOMNO = Literal[
    "Fp1",
    "Fp2",
    "C3",
    "C4",
    "O1",
    "O2",
    "F3",
    "F4",
    "EOG1",
    "EOG2",
    "ECG II",
    "Accu",
    "Accu Stan",
    "SpO2",
    "Pulse",
    "Pleth",
    "Druck Flow",
    "Druck Snore",
    "SpO2",
    "RIP Abdom",
    "RIP Thora",
    "PLMr",
    "PLMl",
    "EMG1",
    "EMG2",
    "EMG3",
    "ArmLi",
    "ArmRe",
    "Schnarch.",
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
    "EOG2:M1",
    "EOG1:M2",
    "EOG2:M2",
    "F4:M1",
    "F3:M2",
    "C4:M1",
    "C3:M2",
    "O2:M1",
    "O1:M2",
    "Fp2:M1",
    "Fp1:M2",
    "Summe RIP",
    "Pos.",
    "Light",
    "Activity",
    "Sync.",
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

PSG_GROUP = Literal["FullPSG", "EEG", "ECG", "EOG", "SpO2", "System", "Resp", "EMG", "Sync", "Activity", "Position"]

PSG_GROUP_MAPPING_SOMNO: Dict[PSG_GROUP, Sequence[PSG_CHANNELS_SOMNO]] = {
    "FullPSG": get_args(PSG_CHANNELS_SOMNO),
    "EEG": [
        "Fp1",
        "Fp2",
        "C3",
        "C4",
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
        "EOG1",
        "EOG2",
        "EOG2:M1",
        "EOG1:M2",
        "EOG2:M2",
        "F4:M1",
        "F3:M2",
        "C4:M1",
        "C3:M2",
        "O2:M1",
        "O1:M2",
        "Fp2:M1",
        "Fp1:M2",
    ],
    "ECG": ["ECG II", "Pulse"],
    "EOG": ["EOG1", "EOG2", "EOG2:M1", "EOG1:M2", "EOG2:M2"],
    "SpO2": ["SpO2"],
    "Sync": ["Sync."],
    "Activity": ["Activity"],
    "System": ["Akku", "Akku Stan", "Light"],
    "Position": ["Pos."],
    "Resp": ["Pleth", "Druck Flow", "RIP Abdom", "RIP Thora", "Summe RIP", "Druck Snore", "Schnarch."],
    "EMG": ["PLMr", "PLMl", "EMG1", "EMG2", "EMG3", "ArmLi", "ArmRe"],
}

PSG_GROUP_MAPPING_MESA: Dict[PSG_GROUP, Sequence[PSG_CHANNELS_MESA]] = {
    "FullPSG": [get_args(PSG_CHANNELS_MESA)],
    "EEG": [
        "EEG1",
        "EEG2",
        "EEG3",
        "EEG1_Off",
        "EEG2_Off",
        "EEG3_Off",
    ],  # Fz, Cz  # Cz, Oz  # C4, M1
    "ECG": [
        "EKGEKG_Off",
        "HR",
        "DHR",
    ],
    "EOG": [
        "EOG-L",
        "EOG-R",
        "EOG-L_Off",
        "EOG-R_Off",
    ],
    "SpO2": [
        "SpO2",
        "OxStatus",
    ],
    "System": ["Therm", "Pos"],
    "Resp": [
        "Flow",
        "Snore",
        "Thor",
        "Abdo",
        "Pleth",
    ],
    "EMG": [
        "EMG",
        "Leg",
        "EMG_Off",
    ],  # Left & Right Leg EMG
}

PSG_CHANNELS_PD_SLEEP_LAB = Literal[
    "Flattening",
    "Right Leg",
    "EKG",
    "Nasendruck",
    "Left Leg",
    "Snore",
    "M1",
    "C3",
    "C3-M2",
    "M2",
    "C4",
    "C4-M1",
    "O2",
    "O2-M1",
    "EOG-L",
    "Left-M1",
    "EOG-R",
    "Right-M1",
    "O1",
    "ChinA",
    "ChinR",
    "Lower.Right-Uppe",
    "ChinL",
    "Lower.Left-Lower",
    "Thorax",
    "Abdomen",
    "Thermistor",
    "Position",
    "F4",
    "F4-M1",
    "pcO2",
    "SpO2",
    "SpO2 BB",
    "Plethysmogram",
    "Pulse",
    "RD-Quality",
    "Herzfrequenz_DR",
    "SpO2-Qualität_DR",
    "Atemfluss_DR",
    "Phase_DR",
    "RMI_DR",
    "RR_DR",
    "XSum_DR",
    "Atemzugsvolumen_",
    "XFlow_DR",
]

PSG_GROUP_MAPPING_PD_SLEEP_LAB: Dict[PSG_GROUP, Sequence[PSG_CHANNELS_PD_SLEEP_LAB]] = {
    "FullPSG": get_args(PSG_CHANNELS_PD_SLEEP_LAB),
    "EEG": [
        "C3",
        "C4",
        "O1",
        "O2",
        "F4",
        "F4-M1",
        "F3",
        "F3-M2",
    ],
    "ECG": ["EKG", "Pulse"],
    "EOG": ["EOG-L", "EOG-R"],
    "SpO2": ["SpO2"],
    "System": ["Thermistor", "Position"],
    "Resp": ["Nasendruck", "Thorax", "Abdomen", "Atemfluss_DR", "Atemzugsvolumen_", "XFlow_DR"],
    "EMG": ["Right Leg", "Left Leg", "ChinA", "ChinR", "ChinL", "Lower.Right-Uppe", "Lower.Left-Lower"],
}


def get_full_PSG(system: PSG_SYSTEM) -> Sequence[str]:
    """Return full PSG."""
    if system not in get_args(PSG_SYSTEM):
        raise ValueError(f"Invalid 'system'! Expected one of {get_args(PSG_SYSTEM)}, got {system}.")
    if system == "mesa":
        return get_args(PSG_CHANNELS_MESA)
    return get_args(PSG_CHANNELS_SOMNO)


def get_psg_channels_by_group(
    system: str,
    psg_channel_group: str,
) -> Sequence[str]:
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
    if system == "somno":
        return PSG_GROUP_MAPPING_SOMNO[psg_channel_group]
    return PSG_GROUP_MAPPING_PD_SLEEP_LAB[psg_channel_group]
