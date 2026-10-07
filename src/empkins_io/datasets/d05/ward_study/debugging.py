import pandas as pd

def get_outside_radar_entries(
    tpcp_dataset,
    radar_windows,
    time_duration=None,
):
    lower_cut_off = "radar_start_time"
    upper_cut_off = "radar_end_time"

    # used to filter unverified rows
    cols_to_check = [
        col for col in tpcp_dataset.COMPLETED_COLS
        if col != "record_id"
    ]

    results = []

    for record_id, participant_windows in radar_windows.groupby("record_id"):
        participant_data = tpcp_dataset.raw_data[
            tpcp_dataset.raw_data["record_id"] == record_id
        ]

        for instrument, properties in tpcp_dataset.TIME_PROPERTIES.items():
            instrument_properties = list(properties.values())

            # Rows belonging to this instrument:
            # at least one instrument property has a value
            instrument_mask = (
                participant_data[instrument_properties]
                .notna()
                .any(axis=1)
            )

            instrument_data = participant_data.loc[
                instrument_mask
            ]

            # Map raw-data index -> row number within this
            # participant + instrument
            instrument_row_map = {
                raw_idx: instrument_row
                for instrument_row, raw_idx
                in enumerate(instrument_data.index, start=1)
            }

            for prop in instrument_properties:

                values = pd.to_datetime(
                    participant_data[prop],
                    errors="coerce",
                ).dropna()

                if values.empty:
                    continue

                within_any_window = pd.Series(
                    False,
                    index=values.index,
                )

                for _, window in participant_windows.iterrows():
                    within_any_window |= values.between(
                        window[lower_cut_off],
                        window[upper_cut_off],
                        inclusive="both",
                    )

                outside_values = values[~within_any_window]

                for idx, actual_time in outside_values.items():

                    candidates = []

                    for _, window in participant_windows.iterrows():
                        radar_start = window[lower_cut_off]
                        radar_end = window[upper_cut_off]

                        distance_to_start = abs(
                            actual_time - radar_start
                        )
                        distance_to_end = abs(
                            actual_time - radar_end
                        )

                        if distance_to_start <= distance_to_end:
                            nearest_boundary = "start"
                            nearest_time = radar_start
                            duration = distance_to_start
                        else:
                            nearest_boundary = "end"
                            nearest_time = radar_end
                            duration = distance_to_end

                        candidates.append({
                            "radar_start": radar_start,
                            "radar_end": radar_end,
                            "nearest_boundary": nearest_boundary,
                            "nearest_radar_time": nearest_time,
                            "duration": duration,
                        })

                    nearest = min(
                        candidates,
                        key=lambda x: x["duration"],
                    )

                    if time_duration is not None:
                        if nearest["duration"] < time_duration:
                            continue

                    is_unverified = (
                        tpcp_dataset.raw_data.loc[idx, cols_to_check] == 1.0
                    ).any()

                    if is_unverified:
                        continue

                    if nearest["duration"] < pd.Timedelta(days=30):
                        reason = "Wrong day?"
                    elif pd.Timedelta(days=30) <= nearest["duration"] <= pd.Timedelta(days=90):
                        reason = "Wrong month?"
                    elif nearest["duration"] > pd.Timedelta(days=90):
                        reason = "Wrong year?"

                    results.append({
                        "record_id": record_id,
                        "instrument": instrument,
                        "property": prop,

                        # Index in complete raw_data
                        "raw_row_index": idx,

                        # Index within participant + instrument
                        "instrument_row": instrument_row_map[idx],

                        "actual_time": actual_time,
                        "radar_start": nearest["radar_start"],
                        "radar_end": nearest["radar_end"],
                        "nearest_boundary": nearest["nearest_boundary"],
                        "nearest_radar_time": nearest["nearest_radar_time"],
                        "duration": nearest["duration"],
                        "suggested_issue": reason
                    })

    return pd.DataFrame(results)

"""
Remove regular_medication entries which start and end within 14 days before radar measurement start. Also remove regular_medication_start entries which are within 10 days before the radar measurement or regular_medication_end entries which are within 5 days after radar end. 
"""
def filter_regular_medication_entries(
    results: pd.DataFrame,
    tpcp_dataset,
    radar_windows: pd.DataFrame,
    regular_med_start_tolerance=pd.Timedelta(days=10),
    regular_med_end_tolerance=pd.Timedelta(days=5),
    regular_med_both_before_tolerance=pd.Timedelta(days=14),
):
    lower_cut_off = "radar_start_time"
    upper_cut_off = "radar_end_time"

    raw_data = tpcp_dataset.raw_data

    regular_med_properties = {
        "regular_med_time_start",
        "regular_med_time_end",
    }

    # Start with the original 7-day result unchanged
    keep_mask = pd.Series(
        True,
        index=results.index,
    )

    removed_entries = []

    # Only inspect regular medication entries that actually appeared
    # in the original 7-day analysis
    regular_med_results = results[
        results["property"].isin(regular_med_properties)
    ]

    # One medication entry is identified by participant + instrument row
    for (record_id, instrument_row), group in regular_med_results.groupby(
        ["record_id", "instrument_row"]
    ):
        # All result rows for this medication entry
        group_indices = group.index

        # All rows should refer to the same raw-data row
        raw_row_index = group["raw_row_index"].iloc[0]

        # Get BOTH start and end from the original raw data.
        # This is important because one of them may already lie inside
        # radar and therefore may not appear in `results`.
        med_start = pd.to_datetime(
            raw_data.loc[
                raw_row_index,
                "regular_med_time_start",
            ],
            errors="coerce",
        )

        med_end = pd.to_datetime(
            raw_data.loc[
                raw_row_index,
                "regular_med_time_end",
            ],
            errors="coerce",
        )

        # We need both values to judge the medication interval.
        if pd.isna(med_start) or pd.isna(med_end):
            continue

        # Clearly invalid interval -> leave it questionable
        if med_end < med_start:
            continue

        participant_windows = radar_windows[
            radar_windows["record_id"] == record_id
        ]

        tolerated = False
        removal_reason = None
        matched_radar_start = None
        matched_radar_end = None

        for _, window in participant_windows.iterrows():
            radar_start = window[lower_cut_off]
            radar_end = window[upper_cut_off]

            # ========================================================
            # CASE 1
            #
            # BOTH medication start and end happen before radar.
            #
            # They are acceptable if BOTH timestamps lie within,
            # for example, 14 days before radar start.
            # ========================================================

            both_before_radar = (
                med_start < radar_start
                and med_end < radar_start
            )

            both_within_tolerance = (
                radar_start - med_start
                <= regular_med_both_before_tolerance
                and radar_start - med_end
                <= regular_med_both_before_tolerance
            )

            if (
                both_before_radar
                and both_within_tolerance
            ):
                tolerated = True

                removal_reason = (
                    "Both medication start and end are before "
                    "radar and within the allowed before-radar "
                    "tolerance."
                )

                matched_radar_start = radar_start
                matched_radar_end = radar_end

                break

            # ========================================================
            # CASE 2
            #
            # Medication overlaps the radar measurement period.
            #
            # Start may be:
            #   - shortly before radar
            #   - inside radar
            #
            # End may be:
            #   - inside radar
            #   - shortly after radar
            #
            # We already checked:
            # med_end >= med_start
            # ========================================================

            start_is_acceptable = (
                # Start shortly BEFORE radar
                (
                    med_start < radar_start
                    and radar_start - med_start <= regular_med_start_tolerance
                )
                # OR start DURING radar
                or (
                    radar_start <= med_start <= radar_end
                )
            )

            end_is_acceptable = (
                (
                    radar_start
                    <= med_end
                    <= radar_end
                )
                or (
                    med_end > radar_end
                    and med_end - radar_end
                    <= regular_med_end_tolerance
                )
            )

            if (
                start_is_acceptable
                and end_is_acceptable
            ):
                tolerated = True

                removal_reason = (
                    "Medication overlaps radar within the "
                    "allowed start/end tolerances."
                )

                matched_radar_start = radar_start
                matched_radar_end = radar_end

                break

        # ============================================================
        # Remove tolerated entries from the original results
        # ============================================================

        if tolerated:
            keep_mask.loc[group_indices] = False

            # Save each removed result row for sanity checking
            for result_idx, result_row in group.iterrows():
                removed_entries.append({
                    "record_id": record_id,
                    "instrument": result_row["instrument"],
                    "instrument_row": instrument_row,
                    "property": result_row["property"],
                    "raw_row_index": raw_row_index,
                    "actual_time": result_row["actual_time"],
                    "regular_med_time_start": med_start,
                    "regular_med_time_end": med_end,
                    "radar_start": matched_radar_start,
                    "radar_end": matched_radar_end,
                    "duration": result_row["duration"],
                    "removal_reason": removal_reason,
                })

    filtered_results = (
        results.loc[keep_mask]
        .reset_index(drop=True)
    )

    removed_regular_med = pd.DataFrame(
        removed_entries
    )

    return filtered_results, removed_regular_med


def export_outside_radar_entries_to_excel(
    results: pd.DataFrame,
    output_path: str,
) -> None:
    # Keep all rows of one participant together,
    # but preserve the order within each participant
    export_results = (
        results
        .sort_values(
            "record_id",
            kind="stable",
        )
        .reset_index(drop=True)
        .copy()
    )

    # Keep duration readable as text:
    # "0 days 05:30:00"
    # "2 days 05:30:00"
    if "duration" in export_results.columns:
        export_results["duration"] = (
            export_results["duration"]
            .astype(str)
        )

    # Datetime columns that may exist in either dataframe
    datetime_columns = {
        "actual_time",
        "radar_start",
        "radar_end",
        "nearest_radar_time",
        "regular_med_time_start",
        "regular_med_time_end",
    }

    with pd.ExcelWriter(
        output_path,
        engine="xlsxwriter",
    ) as writer:

        export_results.to_excel(
            writer,
            sheet_name="Outside Radar",
            index=False,
        )

        workbook = writer.book
        worksheet = writer.sheets["Outside Radar"]

        # ============================================================
        # Basic formats
        # ============================================================

        blue_format = workbook.add_format({
            "bg_color": "#DDEBF7",
        })

        white_format = workbook.add_format({
            "bg_color": "#FFFFFF",
        })

        blue_datetime_format = workbook.add_format({
            "bg_color": "#DDEBF7",
            "num_format": "yyyy-mm-dd hh:mm:ss",
        })

        white_datetime_format = workbook.add_format({
            "bg_color": "#FFFFFF",
            "num_format": "yyyy-mm-dd hh:mm:ss",
        })

        # Merged record_id formats
        blue_record_format = workbook.add_format({
            "bg_color": "#DDEBF7",
            "align": "center",
            "valign": "vcenter",
            "bold": True,
        })

        white_record_format = workbook.add_format({
            "bg_color": "#FFFFFF",
            "align": "center",
            "valign": "vcenter",
            "bold": True,
        })

        # ============================================================
        # Special highlight formats
        # ============================================================

        yellow_format = workbook.add_format({
            "bg_color": "#FFF2CC",
        })

        orange_format = workbook.add_format({
            "bg_color": "#F4B183",
        })

        red_format = workbook.add_format({
            "bg_color": "#F8696B",
        })

        green_format = workbook.add_format({
            "bg_color": "#C6EFCE",
        })

        # ============================================================
        # Write all rows with alternating participant colors
        # ============================================================

        current_record_id = None
        use_blue = False

        for row_idx, row in export_results.iterrows():
            record_id = row["record_id"]

            if record_id != current_record_id:
                use_blue = not use_blue
                current_record_id = record_id

            # Excel row 0 = header
            excel_row = row_idx + 1

            for col_idx, (column, value) in enumerate(row.items()):

                # record_id is merged later
                if column == "record_id":
                    continue

                # Datetime columns
                if column in datetime_columns:
                    if pd.isna(value):
                        worksheet.write_blank(
                            excel_row,
                            col_idx,
                            None,
                            blue_format if use_blue else white_format,
                        )
                    else:
                        # Ensure timestamp is a pandas Timestamp
                        value = pd.to_datetime(value)

                        worksheet.write_datetime(
                            excel_row,
                            col_idx,
                            value.to_pydatetime(),
                            (
                                blue_datetime_format
                                if use_blue
                                else white_datetime_format
                            ),
                        )

                # Everything else
                else:
                    if pd.isna(value):
                        worksheet.write_blank(
                            excel_row,
                            col_idx,
                            None,
                            blue_format if use_blue else white_format,
                        )
                    else:
                        worksheet.write(
                            excel_row,
                            col_idx,
                            value,
                            blue_format if use_blue else white_format,
                        )

        # ============================================================
        # Merge record_id cells per participant
        # ============================================================

        if not export_results.empty:
            record_col = export_results.columns.get_loc("record_id")

            start_row = 1
            current_record_id = export_results.loc[0, "record_id"]
            use_blue = True

            for i in range(1, len(export_results) + 1):
                is_last_row = i == len(export_results)

                if is_last_row:
                    next_record_id = None
                else:
                    next_record_id = export_results.loc[i, "record_id"]

                if is_last_row or next_record_id != current_record_id:
                    end_row = i

                    merge_format = (
                        blue_record_format
                        if use_blue
                        else white_record_format
                    )

                    if start_row == end_row:
                        worksheet.write(
                            start_row,
                            record_col,
                            current_record_id,
                            merge_format,
                        )
                    else:
                        worksheet.merge_range(
                            start_row,
                            record_col,
                            end_row,
                            record_col,
                            current_record_id,
                            merge_format,
                        )

                    if not is_last_row:
                        start_row = i + 1
                        current_record_id = next_record_id
                        use_blue = not use_blue

        # ============================================================
        # Column widths
        # ============================================================

        for col_idx, column in enumerate(export_results.columns):

            if column == "record_id":
                worksheet.set_column(
                    col_idx,
                    col_idx,
                    12,
                )

            elif column in datetime_columns:
                worksheet.set_column(
                    col_idx,
                    col_idx,
                    21,
                )

            elif column == "duration":
                worksheet.set_column(
                    col_idx,
                    col_idx,
                    22,
                )

            else:
                max_length = max(
                    len(str(column)),
                    export_results[column]
                    .astype(str)
                    .str.len()
                    .max(),
                )

                worksheet.set_column(
                    col_idx,
                    col_idx,
                    min(max_length + 2, 50),
                )

        # ============================================================
        # Conditional formatting
        # ============================================================

        if not export_results.empty:
            first_row = 1
            last_row = len(export_results)

            # --------------------------------------------------------
            # suggested_issue only exists in normal filtered results
            # --------------------------------------------------------
            if "suggested_issue" in export_results.columns:
                issue_col = export_results.columns.get_loc(
                    "suggested_issue"
                )

                # Wrong day -> yellow
                worksheet.conditional_format(
                    first_row,
                    issue_col,
                    last_row,
                    issue_col,
                    {
                        "type": "text",
                        "criteria": "containing",
                        "value": "Wrong day?",
                        "format": yellow_format,
                    },
                )

                # Wrong month -> orange
                worksheet.conditional_format(
                    first_row,
                    issue_col,
                    last_row,
                    issue_col,
                    {
                        "type": "text",
                        "criteria": "containing",
                        "value": "Wrong month?",
                        "format": orange_format,
                    },
                )

                # Wrong year -> red
                worksheet.conditional_format(
                    first_row,
                    issue_col,
                    last_row,
                    issue_col,
                    {
                        "type": "text",
                        "criteria": "containing",
                        "value": "Wrong year?",
                        "format": red_format,
                    },
                )

            # --------------------------------------------------------
            # property exists in both result types
            # --------------------------------------------------------
            if "property" in export_results.columns:
                property_col = export_results.columns.get_loc(
                    "property"
                )

                # regular_med_time_start -> green
                worksheet.conditional_format(
                    first_row,
                    property_col,
                    last_row,
                    property_col,
                    {
                        "type": "text",
                        "criteria": "containing",
                        "value": "regular_med_time_start",
                        "format": green_format,
                    },
                )

        # ============================================================
        # Usability
        # ============================================================

        worksheet.freeze_panes(1, 0)

        worksheet.autofilter(
            0,
            0,
            len(export_results),
            len(export_results.columns) - 1,
        )


def export_dataframes_to_excel(
    dataframes: dict[str, pd.DataFrame],
    output_path: str,
) -> None:
    with pd.ExcelWriter(
        output_path,
        engine="xlsxwriter",
        datetime_format="yyyy-mm-dd hh:mm:ss",
    ) as writer:

        workbook = writer.book

        datetime_format = workbook.add_format({
            "num_format": "yyyy-mm-dd hh:mm:ss",
        })

        for sheet_name, df in dataframes.items():
            df = df.copy()

            df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

            worksheet = writer.sheets[sheet_name]

            # Format columns based on dtype
            for col_idx, col in enumerate(df.columns):

                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    worksheet.set_column(
                        col_idx,
                        col_idx,
                        21,
                        datetime_format,
                    )

                else:
                    # Approximate automatic width
                    max_length = max(
                        len(str(col)),
                        df[col]
                        .astype(str)
                        .str.len()
                        .max()
                        if not df.empty
                        else 0,
                    )

                    worksheet.set_column(
                        col_idx,
                        col_idx,
                        min(max_length + 2, 40),
                    )

            worksheet.freeze_panes(1, 0)
