import sqlite3

import numpy as np
import pandas as pd

import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence

import re
import json


class EmradPatientDbHandler:

    patient_id: str
    dir_path: Path
    output_dir_path: Path
    study_start: str
    study_end: str
    tz: str

    SAMPLING_RATE_HZ = 8000000 / 4096 / 2
    N_SAMPLES_PER_PACKET = 32
    MAX_GAP_BETWEEN_PACKETS_S = 0.5
    MAX_GAP_BETWEEN_PACKETS_SEQ = 5
    MAX_N_SEGMENTS = 100
    MIN_SEGMENT_DURATION = 2
    MAX_DURATION_DIFF_S = 10
    MAX_N_PACKETS_PER_DB = 1500000

    def __init__(
            self,
            patient_id: str,
            dir_path: str,
            study_start: Optional[str] = "2024-09-01 00:00:00",
            study_end: Optional[str] = "2026-06-30 23:59:59",
            tz: Optional[str] = None,
    ):
        self.patient_id = patient_id
        self.dir_path = Path(dir_path)
        self.study_start = study_start
        self.study_end = study_end
        self.tz = tz

        if not self.dir_path.is_dir():
            raise ValueError(f"The provided path {self.dir_path} is not a directory.")

        self.output_dir_path = self.dir_path.parent.joinpath("db_segments")


    def create_patient_data_segments(
            self,
    ):

        # TODO add plotting of data overview and phase segmentations

        # get all .db files in the directory
        db_files = list(self.dir_path.glob("*.db"))
        db_files.sort(
            key=lambda x: int(re.search(r"_(\d+)\.db", x.name).group(1))
        )  # sort db files by last number of file name

        if not db_files:
            raise ValueError(f"No .db files found in the directory {self.dir_path}.")

        # create db mapping dict with file names
        self.db_mapping = {f"db_{i}": file.name for i, file in enumerate(db_files)}

        # create db_mapping dict with file paths
        self.db_mapping_paths = {f"db_{i}": file for i, file in enumerate(db_files)}

        # create meta data dict
        self.meta_data_dict = {f"db_{i}": self._check_consistency_of_db_file(file) for i, file in enumerate(db_files)}

        # create meta data dataframe for the whole patient recording
        self.meta_data = self._create_patient_meta_data()

        # create data segments
        self.indices, self.segments = self._create_data_segments()

        # split segments into blocks and store data as db files
        self.segments_overview = []
        self._store_segments_as_db()

    def _check_consistency_of_db_file(self, file):

        connection = sqlite3.connect(file, check_same_thread=False)
        c = connection.cursor()

        c.execute("SELECT rowid, * FROM measurements WHERE processed = 0")
        measurement_rows = c.fetchall()

        if len(measurement_rows) != 1:
            raise ValueError(f"Expected exactly one measurement, but found {len(measurement_rows)}.")

        measurement_id = measurement_rows[0][1]
        comment = measurement_rows[0][2]
        sensor_id = measurement_rows[0][3]
        start = measurement_rows[0][4]
        stop = measurement_rows[0][5]

        print("Got ID:", measurement_id, "Comment:", comment)

        # check if the measurement is within the study period
        study_start_utc = pd.to_datetime(self.study_start).tz_localize(self.tz).timestamp()
        study_end_utc = pd.to_datetime(self.study_end).tz_localize(self.tz).timestamp()

        if not (study_start_utc <= start <= study_end_utc):
            raise ValueError(
                f"Measurement start time {start} is not within the study period {self.study_start} - {self.study_end}.")

        if not (study_start_utc <= stop <= study_end_utc):
            raise ValueError(
                f"Measurement stop time {stop} is not within the study period {self.study_start} - {self.study_end}.")

        # check general settings
        c.execute("SELECT rowid, timestamp, sensor_id, sequence_id, uptime, data_format, data_size FROM packets")
        data = c.fetchall()
        data = np.array(data)

        if not (np.any(start <= data[:, 1]) & np.any(data[:, 1] <= stop)):  # timestamp
            raise ValueError(f"Measurement timestamps are not within the range {start} - {stop}.")

        if not ([sensor_id] == list(np.unique(data[:, 2]))):  # sensor_id
            raise ValueError(
                f"Sensor ID {sensor_id} does not match the sensor IDs in the data: {list(np.unique(data[:, 2]))}.")

        if len(np.unique(data[:, 5])) != 1:  # data_format
            raise ValueError(f"Data format is not consistent: {np.unique(data[:, 5])}.")

        if len(np.unique(data[:, 6])) != 1:  # data_size
            raise ValueError(f"Data size is not consistent: {np.unique(data[:, 6])}.")

        meta_data = pd.DataFrame(
            data,
            columns=["db_index", "timestamp", "sensor_id", "sequence_id", "uptime", "data_format", "data_size"]
        )

        return meta_data

    def _create_patient_meta_data(self):

        dbs = list(self.db_mapping.keys())
        data_cols = list(self.meta_data_dict["db_0"].columns)

        # create list of column names
        cols = dbs.copy()
        cols.append("db")
        cols.extend(data_cols)

        # create dataframe for whole patient
        df_meta_data = pd.DataFrame(columns=cols)
        df_meta_data[dbs] = df_meta_data[dbs].astype(bool)

        # loop over the included databases
        for db_name, df_db in self.meta_data_dict.items():
            df_tmp = pd.DataFrame(columns=cols)
            df_tmp[data_cols] = df_db[data_cols]
            df_tmp[dbs] = df_tmp[dbs].astype(bool)
            df_tmp[dbs] = False
            df_tmp[db_name] = True
            df_tmp["db"] = db_name
            df_meta_data = pd.concat([df_meta_data, df_tmp], axis=0)

        df_meta_data = df_meta_data.reset_index(drop=True) # reset index since the index is not unique

        return df_meta_data

    def _create_data_segments(self):

        data_sorted = self.meta_data.copy()
        data_sorted = data_sorted.sort_values(by="timestamp").reset_index(drop=True)

        # identify measurement segments based on timestamps
        indices_ts = np.where(np.diff(data_sorted["timestamp"].to_numpy()) > self.MAX_GAP_BETWEEN_PACKETS_S)[0]

        # identify measurement segments based on sequence_id
        indices_sq = np.where(np.abs(np.diff(data_sorted["sequence_id"].to_numpy())) > self.MAX_GAP_BETWEEN_PACKETS_SEQ)[0]

        if not (np.any(indices_ts == indices_sq)):
            raise ValueError("Indices based on timestamps and sequence IDs do not match.")

        indices = indices_ts + 1

        if len(indices) > self.MAX_N_SEGMENTS:
            warnings.warn(
                f"Number of measurement segments ({len(indices)}) exceeds the maximum allowed ({self.MAX_N_SEGMENTS}).")

        indices = np.insert(indices, 0, 0)  # set the first index to 0
        indices = np.append(indices, len(data_sorted))  # set the last index to the length of the data

        # create data segments
        packet_segments = {}
        for i in range(len(indices) - 1):
            start_idx = indices[i]
            stop_idx = indices[i + 1]

            if not (start_idx <= stop_idx):
                raise ValueError(f"Start index {start_idx} is not less than stop index {stop_idx} for segment {i}.")

            packet_segment = data_sorted.iloc[start_idx:stop_idx, :]
            packet_segment_sorted = packet_segment.sort_values(by="sequence_id").reset_index()
            packet_segment_sorted = packet_segment_sorted.rename(columns={"index": "patient_index"})

            if self._segment_check(packet_segment_sorted):
                packet_segments[f"segment_{i}"] = packet_segment_sorted
            else:
                warnings.warn(f"segment_{i} does not meet the criteria")
                packet_segments[f"segment_{i}"] = np.nan

        return indices, packet_segments

    def _segment_check(self, packet_segment):

        # check if the segment is empty
        if len(packet_segment) == 0:
            return False

        # check for duplicate sequence IDs
        if packet_segment["sequence_id"].duplicated().any():
            raise ValueError("Duplicate sequence IDs found in the meta data.")

        # check for continuous sequence IDs
        if not np.all(np.diff(packet_segment["sequence_id"]) == 1):
            raise ValueError("Sequence IDs are not continuous.")

        # check for minimum segment duration of 2 seconds
        if (packet_segment.timestamp.max() - packet_segment.timestamp.min()) < self.MIN_SEGMENT_DURATION:
            raise ValueError("Segment duration is less than 2 seconds.")

        # check for duration differences
        expected_duration = len(packet_segment) * self.N_SAMPLES_PER_PACKET / self.SAMPLING_RATE_HZ
        measured_duration = (packet_segment.timestamp.max() - packet_segment.timestamp.min())
        if np.abs(measured_duration-expected_duration) > self.MAX_DURATION_DIFF_S:
            raise ValueError(
                f"The difference between expected and "
                f"measured duration is {abs(expected_duration - measured_duration):.2f} s."
            )

        return True

    def _store_segments_as_db(self):

        # example of naming of databases: EMP_1001_segment_5_block_4

        self.output_dir_path.mkdir(exist_ok=True)  # create output db directory

        # loop through all segments
        for segment_id, df_segment in self.segments.items():

            print("Segment: ", segment_id)

            # create data blocks per segment
            blocks = {
                f"block_{i}": df_segment.iloc[i * self.MAX_N_PACKETS_PER_DB: (i + 1) * self.MAX_N_PACKETS_PER_DB]
                for i in range((len(df_segment) + self.MAX_N_PACKETS_PER_DB - 1) // self.MAX_N_PACKETS_PER_DB)
            }

            # loop through blocks of segment
            for block_id, df_block in blocks.items():
                db_block_path = self.output_dir_path.joinpath(f"{self.patient_id}_{segment_id}_{block_id}.db")

                self._add_row_to_data_overview(
                    segment_id=segment_id,
                    block_id=block_id,
                    start_utc=df_block["timestamp"].min(),
                    end_utc=df_block["timestamp"].max(),
                    sensor_id=int(df_block["sensor_id"].unique()[0])
                )

                self._add_meta_data_table_to_db(
                    db_path=db_block_path,
                    segment_id=segment_id,
                    block_id=block_id,
                    block_start=df_block["timestamp"].min(),
                    block_end=df_block["timestamp"].max(),
                    n_blocks_in_segment=len(blocks.keys()),
                    sensor_id=int(df_block["sensor_id"].unique()[0])
                )

                self._add_packet_meta_data_to_db(
                    db_path=db_block_path,
                    df_block=df_block
                )

                for db_in in list(df_block["db"].unique()):
                    self._add_data_packets_to_db(
                        db_path_in=self.db_mapping_paths[db_in],
                        db_path_out=db_block_path,
                        db_in=db_in
                    )

        pd.DataFrame(self.segments_overview).to_excel(
                self.output_dir_path.joinpath(f"{self.patient_id}_data_overview.xlsx")
        )

    def _add_row_to_data_overview(
            self, segment_id: str, block_id: str, start_utc: float, end_utc: float, sensor_id: int
    ):

        start_ts = pd.to_datetime(start_utc, unit='s', utc=True).tz_convert(self.tz)
        end_ts = pd.to_datetime(end_utc, unit='s', utc=True).tz_convert(self.tz)

        self.segments_overview.append(
            {
                "patient_id": self.patient_id,
                "sensor_id": sensor_id,
                "segment_id": segment_id,
                "block_id": block_id,
                "start_utc": start_utc,
                "start_date (Europe/Berlin)": str(start_ts.date()),
                "start_time (Europe/Berlin)": str(start_ts.time()),
                "end_utc": end_utc,
                "end_date (Europe/Berlin)": str(end_ts.date()),
                "end_time (Europe/Berlin)": str(end_ts.time()),
            }
        )

    def _add_meta_data_table_to_db(
            self, db_path, segment_id, block_id, block_start, block_end, n_blocks_in_segment, sensor_id
    ):

        # Connect to database
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # cur.execute("DROP TABLE IF EXISTS meta_data;")

        # Step 1: Create meta_data table if it doesn't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS meta_data (
                patient_id TEXT,
                segment_id TEXT,
                block_id TEXT,
                block_start REAL,
                block_end REAL,
                n_blocks_in_segment INTEGER,
                sensor_id INTEGER,
                db_mapping TEXT
            )
        """)

        # Step 2: Insert metadata
        cur.execute("""
            INSERT INTO meta_data (patient_id, segment_id, block_id, block_start, block_end, n_blocks_in_segment, sensor_id, db_mapping)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.patient_id,
            segment_id,
            block_id,
            block_start,
            block_end,
            n_blocks_in_segment,
            sensor_id,
            json.dumps(self.db_mapping)  # store dict as JSON
        ))

        conn.commit()
        conn.close()

    def _add_packet_meta_data_to_db(
            self,db_path, df_block
    ):

        db_cols = list(self.db_mapping.keys())
        df_db = df_block.copy()

        df_db = df_db.reset_index()
        df_db = df_db.rename(columns={"index": "segment_index"})

        df_db[db_cols] = df_db[db_cols].astype(int)  # store booleans as 0/1 in SQL
        df_db = df_db.astype({
            "db_index": int,
            "sensor_id": int,
            "sequence_id": int,
            "uptime": int,
            "data_format": int,
            "data_size": int
        })

        conn = sqlite3.connect(db_path)

        df_db.to_sql("packets", conn, if_exists="replace", index=False)

        conn.close()

    def _add_data_packets_to_db(
            self, db_path_in, db_path_out, db_in
    ):

        conn = sqlite3.connect(db_path_out)

        # Step 1: Make sure 'data' column exists in main.packets
        try:
            conn.execute("ALTER TABLE packets ADD COLUMN data BLOB;")
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE packets DROP COLUMN data;")
            conn.execute("ALTER TABLE packets ADD COLUMN data BLOB;")

        conn.execute(f"ATTACH DATABASE '{db_path_in}' AS db_in")

        conn.execute("""
            UPDATE main.packets
            SET data = (
                SELECT db_in.packets.data
                FROM db_in.packets
                WHERE db_in.packets.rowid = main.packets.db_index
                AND main.packets.db = ?
            )
        """, (db_in,))

        conn.commit()
        conn.close()



