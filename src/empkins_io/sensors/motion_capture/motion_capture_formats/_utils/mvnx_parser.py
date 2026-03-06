from xml.etree import ElementTree

import h5py
import numpy as np
from tqdm.auto import tqdm

__all__ = ["_MvnxParser"]


class _MvnxParser:
    """
    The abstract parser object to run through the XML tree structure of the MVNX file format
    and extract the relevant information into dictionaries and numpy arrays. Super simple, needs
    refactoring at the moment.

    Can also be used as a command line tool.
    """

    def __init__(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        path,
        orientation=None,
        position=None,
        velocity=None,
        acceleration=None,
        angular_velocity=None,
        angular_acceleration=None,
        foot_contacts=None,
        sensor_free_acceleration=None,
        sensor_magnetic_field=None,
        sensor_orientation=None,
        joint_angle=None,
        joint_angle_xzy=None,
        joint_angle_ergo=None,
        center_of_mass=None,
        mapping=None,
        sensors=None,
        segments=None,
        joints=None,
        root=None,
        mvn=None,
        comment=None,
        subject=None,
        version=None,
        build=None,
        label=None,
        frame_rate=None,
        segment_count=None,
        recording_date=None,
        configuration=None,
        user_scenario=None,
        security_code=None,
        modality=None,
        time=None,
        index=None,
        timecode=None,
        ms=None,
        *,
        verbose=True,
    ):
        if orientation is None:
            self.orientation = []
        if position is None:
            self.position = []
        if velocity is None:
            self.velocity = []
        if acceleration is None:
            self.acceleration = []
        if angular_velocity is None:
            self.angularVelocity = []
        if angular_acceleration is None:
            self.angularAcceleration = []
        if foot_contacts is None:
            self.footContacts = []
        if sensor_free_acceleration is None:
            self.sensorFreeAcceleration = []
        if sensor_magnetic_field is None:
            self.sensorMagneticField = []
        if sensor_orientation is None:
            self.sensorOrientation = []
        if joint_angle is None:
            self.jointAngle = []
        if joint_angle_xzy is None:
            self.jointAngleXZY = []
        if joint_angle_ergo is None:
            self.jointAngleErgo = []
        if center_of_mass is None:
            self.centerOfMass = []
        if sensors is None:
            self.sensors = []
        if segments is None:
            self.segments = {}
        if joints is None:
            self.joints = {}
        if mapping is None:
            self.mapping = {
                "orientation": 0,
                "position": 1,
                "velocity": 2,
                "acceleration": 3,
                "angularVelocity": 4,
                "angularAcceleration": 5,
                "footContacts": 6,
                "sensorFreeAcceleration": 7,
                "sensorMagneticField": 8,
                "sensorOrientation": 9,
                "jointAngle": 10,
                "jointAngleXZY": 11,
                "jointAngleErgo": 12,
                "jointAngleErgoXZY": 13,
                "centerOfMass": 14,
            }
        if time is None:
            self.time = []
        else:
            self.time = time
        if index is None:
            self.index = []
        else:
            self.index = index
        if timecode is None:
            self.timecode = []
        else:
            self.timecode = timecode
        if ms is None:
            self.ms = []
        else:
            self.ms = ms
        self.mvn = mvn
        self.comment = comment
        self.subject = subject
        self.version = version
        self.build = build
        self.label = label
        self.frameRate = frame_rate
        self.segmentCount = segment_count
        self.recordingDate = recording_date
        self.configuration = configuration
        self.userScenario = user_scenario
        self.securityCode = security_code
        self.modality = modality
        self.verbose = verbose
        if path is None:
            print("Please supply a path")
        self.path = path
        if root is None:
            self.parse_mvnx(self.path)
            self.parse_all()
        else:
            self.root = root

    def __repr__(self):
        return f"<MVNX ({self.path})>"

    def parse_mvnx(self, path):
        """
        Take a path to an MVNX file and parse it.

        Args:
            path ([string]): [the path to the data file]
        """
        tree = ElementTree.parse(path)
        self.root = tree.getroot()
        self.mvn = self.root[0]
        self.version = self.root[0].attrib["version"]
        self.build = self.root[0].attrib["build"]
        self.comment = self.root[1].text
        self.label = self.root[2].attrib["label"]
        self.frameRate = float(self.root[2].attrib["frameRate"])
        self.segmentCount = int(self.root[2].attrib["segmentCount"])
        self.recordingDate = int(self.root[2].attrib["recDateMSecsSinceEpoch"])
        self.configuration = self.root[2].attrib["configuration"]
        self.userScenario = self.root[2].attrib["userScenario"]
        self.securityCode = self.root[3].attrib["code"]
        return self.root

    def parse_modality(self, modality):
        """[With a given XML Tree, parse out the salient modalities within each frame].

        Args:

            modality ([string]): [the name of the modality]

        """
        holding_list = []
        frames = self.root[2][6]
        for frame in frames[3:]:
            for child in frame[self.mapping[modality] : self.mapping[modality] + 1]:
                holding_list.append(child.text.split(" "))
        holding_list = np.array(holding_list)
        return holding_list.astype(float)

    def parse_time(self):
        frames = self.root[2][6][3:]
        for frame in frames:
            self.time.append(frame.attrib["time"])
        return self.time

    def parse_index(self):
        frames = self.root[2][6][3:]
        for frame in frames:
            self.index.append(frame.attrib["index"])
        return self.index

    def parse_timecode(self):
        frames = self.root[2][6][3:]
        for frame in frames:
            self.timecode.append(frame.attrib["tc"])
        return self.timecode

    def parse_ms(self):
        frames = self.root[2][6][3:]
        for frame in frames:
            self.ms.append(frame.attrib["ms"])
        return self.ms

    def parse_modalities(self, *args):
        for arg in args:
            print(self.parse_modality(arg))
            return self.parse_modality(arg)
        return None

    def parse_sensors(self):
        for sensor in self.root[2][2]:
            self.sensors.append(sensor.attrib["label"])
        return self.sensors

    def parse_segments(self):
        for segment in self.root[2][1]:
            self.segments[segment.attrib["id"]] = segment.attrib["label"]
        return self.segments

    def parse_joints(self):
        for joint in self.root[2][3]:
            self.joints[joint.attrib["label"]] = [joint[0].text, joint[1].text]
        return self.joints

    def parse_all(self):
        for key in tqdm(self.mapping.keys(), desc="Parse Channels", disable=not self.verbose):
            setattr(self, key, self.parse_modality(key))
        self.parse_time()
        self.parse_joints()
        self.parse_segments()
        self.parse_sensors()
        self.parse_timecode()
        self.parse_ms()

    def save_to_hdf5(self, filepath):
        """Create an HDF5 file from an MVNX object.

        :param filepath: _description_ - the path to the file you want to create
        :type filepath: _type_
        """
        with h5py.File(f"{filepath}.hdf5", "w") as f:
            f.create_dataset("jointAngle", data=self.jointAngle)
            f.create_dataset("jointAngleXZY", data=self.jointAngleXZY)
            f.create_dataset("angularVelocity", data=self.angularVelocity)
            f.create_dataset("angularAcceleration", data=self.angularAcceleration)
            f.create_dataset("position", data=self.position)
            f.create_dataset("orientation", data=self.orientation)
            f.create_dataset("centerOfMass", data=self.centerOfMass)
