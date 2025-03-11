from struct import unpack

from construct import *
import numpy as np
import sqlite3
import h5py
from biopsykit.utils._datatype_validation_helper import _assert_file_extension

from pathlib import Path


class EmradParser:

    def __init__(self):
        pass

    def parse(self, data):
        try:

            data = np.reshape(unpack('<320i', data), (32, 10))
            return data
        except:
            pass


class EmradFileConverter:
    """Class converting a emRad measurement from db to h5 file format

    Attributes
    ----------
    file_path : str
        path to db file including file extension
    """

    file_path: str

    def __init__(
            self,
            file_path: str
    ):
        self.file_path = file_path

    def convert_db_to_h5(self):
        path = Path(self.file_path)
        _assert_file_extension(self.file_path, ".db")

        parser = EmradParser()

        connection = sqlite3.connect(self.file_path, check_same_thread=False)
        c = connection.cursor()

        c.execute("SELECT rowid, * FROM measurements WHERE processed = 0")
        rows = c.fetchall()

        for row in rows:
            # rowid = row[0]
            measurement_id = row[1]
            comment = row[2]
            sensor_id = int(row[3])
            start = row[4]
            stop = row[5]
            print("Got Measurement ID:", measurement_id)

            c.execute("SELECT * FROM packets WHERE sensor_id = ? and timestamp BETWEEN ? and ?", (sensor_id, start, stop))
            packets = c.fetchall()
            series = []
            for packet in packets:
                series.append(parser.parse(packet[6]))
            try:
                series = np.concatenate(series)
            except:
                series = np.array([])

            hf = h5py.File(path.with_suffix('.h5'), 'w')
            g = hf.create_group('Radar')
            g.attrs["measurement_id"] = measurement_id
            g.attrs["comment"] = comment
            g.attrs["sensor_id"] = sensor_id
            g.attrs["start"] = start
            g.attrs["stop"] = stop

            g.create_dataset('rad1', data=series[:, [0, 1, 8, 9]])
            g.create_dataset('rad2', data=series[:, [2, 3, 8, 9]])
            g.create_dataset('rad3', data=series[:, [4, 5, 8, 9]])
            g.create_dataset('rad4', data=series[:, [6, 7, 8, 9]])

            hf.close()

        connection.close()
