import sqlite3

import numpy as np
import pandas as pd

import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence

import re
import json

import matplotlib.pyplot as plt


class EmradPatientDbHandler:

    patient_id: str
    dir_path: Path
    output_dir_path: Path
    study_start: str
    study_end: str
    tz: str
    plot: bool

    SAMPLING_RATE_HZ = 8000000 / 4096 / 2
    N_SAMPLES_PER_PACKET = 32
    MAX_GAP_BETWEEN_PACKETS_S = 0.5
    MAX_N_SEGMENTS = 100
    MIN_SEGMENT_DURATION = 1
    MAX_N_PACKETS_PER_DB = 3000000

    def __init__(
            self,
            patient_id: str,
            dir_path: str,
            study_start: Optional[str] = "2024-09-01 00:00:00",
            study_end: Optional[str] = "2026-06-30 23:59:59",
            tz: Optional[str] = None,
            plot: Optional[bool] = False
    ):
        self.patient_id = patient_id
        self.dir_path = Path(dir_path)
        self.study_start = study_start
        self.study_end = study_end
        self.tz = tz
        self.plot = plot

        if not self.dir_path.is_dir():
            raise ValueError(f"The provided path {self.dir_path} is not a directory.")

        self.output_dir_path = self.dir_path.parent.joinpath("db_segments")


    def create_patient_data_segments(
            self,
    ):

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
        self.segments = self._create_data_segments()

        # calculate gaps between segments in seconds
        self._calculate_gaps()

        # split segments into blocks and store data as db files
        self.segments_overview = []
        self._store_segments_as_db()

        # plot if flag is set
        if self.plot:
            self._plot()

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

        if not (np.all(start <= data[:, 1]) & np.all(data[:, 1] <= stop)):  # timestamp
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

        # identify duplicate values (can happen, when overlapping database parts are extracted)
        # extract the numeric part from "db" and use it as priority
        df_meta_data["db_priority"] = df_meta_data["db"].str.extract(r"db_(\d+)").astype(int)
        # sort by timestamp, sequence_id, uptime, then db_priority
        df_meta_data = df_meta_data.sort_values(by=["timestamp", "sequence_id", "uptime", "db_priority"])
        # drop duplicates keeping the lowest db_x due to sorting
        df_meta_data = df_meta_data.drop_duplicates(
            subset=["timestamp", "sequence_id", "uptime"],
            keep="first"
        )
        # clean up helper column
        df_meta_data = df_meta_data.drop(columns=["db_priority"])

        # sort data by timestamp, then by sequence_id and uptime an create patient index
        df_meta_data = df_meta_data.sort_values(by=["timestamp", "sequence_id", "uptime"]).reset_index(drop=True)
        df_meta_data = df_meta_data.reset_index()
        df_meta_data = df_meta_data.rename(columns={"index": "patient_index"})

        return df_meta_data

    def _create_data_segments(self):

        data_sorted = self.meta_data.copy()

        # identify measurement segments based on timestamps
        indices_ts = list(data_sorted.index[data_sorted["timestamp"].diff() > self.MAX_GAP_BETWEEN_PACKETS_S])
        indices_ts.insert(0, 0) # set the first index to 0
        indices_ts.append(len(data_sorted)) # set the last index to the length of the data

        if len(indices_ts) > self.MAX_N_SEGMENTS:
            warnings.warn(
                f"Number of measurement segments ({len(indices_ts) - 1}) exceeds the maximum allowed ({self.MAX_N_SEGMENTS}).")

        packet_segments = {}
        segment_cnt = 0

        # create data segments based on timestamps
        for i in range(len(indices_ts) - 1):
            start_idx_ts = indices_ts[i]
            stop_idx_ts = indices_ts[i + 1]

            if not (start_idx_ts <= stop_idx_ts):
                raise ValueError(f"Start index {start_idx_ts} is not less than stop index {stop_idx_ts} for segment {i}.")

            # cut segment and sort by sequence_id
            packet_segment_ts = data_sorted.iloc[start_idx_ts:stop_idx_ts, :]
            packet_segment_ts_sorted = packet_segment_ts.sort_values(by="sequence_id").reset_index(drop=True)

            # identify if sequence_id is continuous
            indices_si = list(packet_segment_ts_sorted.index[packet_segment_ts_sorted["sequence_id"].diff() > 1])
            indices_si.insert(0, 0)  # set the first index to 0
            indices_si.append(len(packet_segment_ts_sorted))  # set the last index to the length of the data

            # create data segments based on sequence_id
            for j in range(len(indices_si) - 1):
                start_idx_si = indices_si[j]
                stop_idx_si = indices_si[j+1]
                packet_segment_si = packet_segment_ts_sorted.iloc[start_idx_si : stop_idx_si, :]

                segment_meta_data = {
                    "start_index": start_idx_ts+start_idx_si,
                    "stop_index": start_idx_ts+stop_idx_si,
                    "n_packets": len(packet_segment_si)
                }

                check_results = self._segment_check(packet_segment_si)
                segment_meta_data.update(check_results)

                packet_segments[segment_cnt] = {
                    "segment_meta_data": segment_meta_data,
                    "data": packet_segment_si.reset_index(drop=True)
                }

                segment_cnt += 1

        return packet_segments

    def _segment_check(self, packet_segment):

        results = {
            "duplicate_sequence_id": False,
            "segment_duration_smaller_1_s": False,
            "discontinuous_sequence_id": False,
        }

        # check for duplicate sequence IDs
        if packet_segment["sequence_id"].duplicated().any():
            results["duplicate_sequence_id"] = True

        # check for minimum segment duration
        min_n_packets = np.floor(self.SAMPLING_RATE_HZ / self.N_SAMPLES_PER_PACKET * self.MIN_SEGMENT_DURATION)
        if len(packet_segment) < min_n_packets:
            results["segment_duration_smaller_1_s"] = True

        # check for continuous sequence IDs
        if not np.all(np.diff(packet_segment["sequence_id"]) == 1):
            results["discontinuous_sequence_id"] = True

        if np.any(list(results.values())):
            results["check_passed"] = False
        else:
            results["check_passed"] = True

        return results

    def _calculate_gaps(self):
        timings = {}

        for key, data in self.segments.items():
            timings[key] = {
                "timestamp_min": data["data"]["timestamp"].min(),
                "timestamp_max": data["data"]["timestamp"].max(),
                "sequence_id_min": data["data"]["sequence_id"].min(),
                "sequence_id_max": data["data"]["sequence_id"].max()
            }

        df_timings = pd.DataFrame(timings).T

        # calculate gap between row i (timestamp_max) and row i+1 (timestamp_min)
        df_timings["timestamp_gap"] = df_timings["timestamp_min"].shift(-1) - df_timings["timestamp_max"]
        df_timings["sequence_id_gap"] = df_timings["sequence_id_min"].shift(-1) - df_timings["sequence_id_max"]

        for index, row in df_timings.iterrows():
            self.segments[index]["segment_meta_data"]["gap_to_following_segment_s"] = row["timestamp_gap"]
            self.segments[index]["segment_meta_data"]["gap_to_following_segment_seq_id"] = row["sequence_id_gap"]


    def _store_segments_as_db(self):

        self.output_dir_path.mkdir(exist_ok=True)  # create output db directory

        # loop through all segments
        for segment_id in self.segments.keys():

            segment_meta_data = self.segments[segment_id]["segment_meta_data"].copy()
            df_segment = self.segments[segment_id]["data"]

            # create data blocks per segment
            blocks = {
                i: df_segment.iloc[i * self.MAX_N_PACKETS_PER_DB: (i + 1) * self.MAX_N_PACKETS_PER_DB]
                for i in range((len(df_segment) + self.MAX_N_PACKETS_PER_DB - 1) // self.MAX_N_PACKETS_PER_DB)
            }

            # loop through blocks of segment
            for block_id, df_block in blocks.items():

                file_name = f"{self.patient_id}_segment_{segment_id:03d}_block_{block_id:03d}.db"
                print(file_name)

                db_block_path = self.output_dir_path.joinpath(file_name)

                self._add_row_to_data_overview(
                    file_name=file_name,
                    segment_id=segment_id,
                    block_id=block_id,
                    gap_to_following_segment_s=segment_meta_data["gap_to_following_segment_s"],
                    gap_to_following_segment_seq=segment_meta_data["gap_to_following_segment_seq_id"],
                    start_utc=df_block["timestamp"].min(),
                    end_utc=df_block["timestamp"].max(),
                    sensor_id=int(df_block["sensor_id"].unique()[0]),
                    db=", ".join(str(db) for db in df_block["db"].unique()),
                    start_index=segment_meta_data["start_index"] + df_block.index[0],
                    stop_index=segment_meta_data["start_index"] + df_block.index[-1],
                    n_packets=len(df_block),
                    segment_check_passed=segment_meta_data["check_passed"],
                    duplicate_sequence_id=segment_meta_data["duplicate_sequence_id"],
                    segment_duration_smaller_1_s=segment_meta_data["segment_duration_smaller_1_s"],
                    discontinuous_sequence_id=segment_meta_data["discontinuous_sequence_id"],
                )

                # if segment_meta_data["check_passed"]:
                #     self._add_meta_data_table_to_db(
                #         db_path=db_block_path,
                #         segment_id=segment_id,
                #         block_id=block_id,
                #         block_start=df_block["timestamp"].min(),
                #         block_end=df_block["timestamp"].max(),
                #         n_blocks_in_segment=len(blocks.keys()),
                #         sensor_id=int(df_block["sensor_id"].unique()[0]),
                #         db=", ".join(str(db) for db in df_block["db"].unique())
                #     )
                #
                #     self._add_packet_meta_data_to_db(
                #         db_path=db_block_path,
                #         df_block=df_block
                #     )
                #
                #     for db_in in list(df_block["db"].unique()):
                #         self._add_data_packets_to_db(
                #             db_path_in=self.db_mapping_paths[db_in],
                #             db_path_out=db_block_path,
                #             db_in=db_in
                #         )

        pd.DataFrame(self.segments_overview).to_excel(
                self.output_dir_path.joinpath(f"{self.patient_id}_data_overview.xlsx")
        )

    def _add_row_to_data_overview(
            self,
            file_name: str,
            segment_id: str,
            block_id: str,
            gap_to_following_segment_s: float,
            gap_to_following_segment_seq: float,
            start_utc: float,
            end_utc: float,
            sensor_id: int,
            db: str,
            start_index: int,
            stop_index: int,
            n_packets: int,
            segment_check_passed: bool,
            duplicate_sequence_id: bool,
            segment_duration_smaller_1_s: bool,
            discontinuous_sequence_id: bool
    ):

        start_ts = pd.to_datetime(start_utc, unit='s', utc=True).tz_convert(self.tz)
        end_ts = pd.to_datetime(end_utc, unit='s', utc=True).tz_convert(self.tz)

        self.segments_overview.append(
            {
                "file_name": file_name,
                "patient_id": self.patient_id,
                "segment_id": segment_id,
                "block_id": block_id,
                "sensor_id": sensor_id,
                "db": db,
                "gap_to_following_segment_s": gap_to_following_segment_s,
                "gap_to_following_segment_seq_id": gap_to_following_segment_seq,
                "start_patient_index": start_index,
                "stop_patient_index": stop_index,
                "n_packets": n_packets,
                "segment_check_passed": segment_check_passed,
                "duplicate_sequence_id": duplicate_sequence_id,
                "segment_duration_smaller_1_s": segment_duration_smaller_1_s,
                "discontinuous_sequence_id": discontinuous_sequence_id,
                "start_utc": start_utc,
                "start_date (Europe/Berlin)": str(start_ts.date()),
                "start_time (Europe/Berlin)": str(start_ts.time()),
                "end_utc": end_utc,
                "end_date (Europe/Berlin)": str(end_ts.date()),
                "end_time (Europe/Berlin)": str(end_ts.time()),
            }
        )

    def _add_meta_data_table_to_db(
            self,
            db_path: Path,
            segment_id: int,
            block_id: int,
            block_start: float,
            block_end: float,
            n_blocks_in_segment: int,
            sensor_id: int,
            db: str
    ):

        # Connect to database
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("DROP TABLE IF EXISTS meta_data;")

        # Step 1: Create meta_data table if it doesn't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS meta_data (
                patient_id TEXT,
                included_dbs TEXT,
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
            INSERT INTO meta_data (patient_id, included_dbs, segment_id, block_id, block_start, block_end, n_blocks_in_segment, sensor_id, db_mapping)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.patient_id,
            db,
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
        cur = conn.cursor()

        # Step 1: Check if column 'data' exists in packets
        cur.execute("PRAGMA table_info(packets);")
        columns = [row[1] for row in cur.fetchall()]  # row[1] is the column name

        if "data" not in columns:
            cur.execute("ALTER TABLE packets ADD COLUMN data BLOB;")

        # Step 2: Attach other database
        cur.execute(f"ATTACH DATABASE '{db_path_in}' AS db_in")
        cur.execute("""
            UPDATE main.packets
            SET data = (
                SELECT db_in.packets.data
                FROM db_in.packets
                WHERE db_in.packets.rowid = main.packets.db_index
                AND main.packets.db = ?
            )
            WHERE data IS NULL
        """, (db_in,))

        conn.commit()
        cur.execute("DETACH DATABASE db_in;")
        conn.close()


    def _plot(self):

        # overview figure
        df_plot = self.meta_data.copy()
        df_plot = df_plot.drop(columns=["patient_index", "db", "data_format", "data_size"])
        df_plot = df_plot.sort_values(by="timestamp").reset_index(drop=True)

        fig, axs = plt.subplots(df_plot.shape[1], figsize=(10, 15), sharex=True)

        df_plot.astype(int).plot(subplots=True, ax=axs, legend="upper_right", xlabel="patient_index")
        fig.suptitle(f"{self.patient_id}_overview")

        fig.tight_layout()
        fig.savefig(self.output_dir_path.joinpath(f"{self.patient_id}_data_overview.pdf"))
        plt.close(fig)

        # segments figure
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].plot(df_plot["timestamp"])
        axs[1].plot(df_plot["sequence_id"])
        axs[1].set_xlabel("patient_index")
        axs[0].set_ylabel("timestamp")
        axs[1].set_ylabel("sequence_id")
        fig.suptitle(f"{self.patient_id}_segments")

        for index, data in self.segments.items():
            start_idx = data["segment_meta_data"]["start_index"]
            stop_idx = data["segment_meta_data"]["stop_index"]

            if data["segment_meta_data"]["check_passed"]:
                axs[0].axvspan(xmin=start_idx, xmax=stop_idx, color="green", alpha=0.3)
                axs[1].axvspan(xmin=start_idx, xmax=stop_idx, color="green", alpha=0.3)
            else:
                axs[0].axvspan(xmin=start_idx, xmax=stop_idx, color="red", alpha=0.5)
                axs[1].axvspan(xmin=start_idx, xmax=stop_idx, color="red", alpha=0.5)

        fig.savefig(self.output_dir_path.joinpath(f"{self.patient_id}_data_segmentation.pdf"))
        plt.close(fig)
