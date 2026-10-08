from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIMEZONE = ZoneInfo("Europe/Berlin")

# channels that carry the synchronization signal and are therefore not plotted in the overview
SYNC_CHANNELS = ("sync", "sync_in", "sync_out")


def activities_overview_plotting(emrad_data, log, subject):
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(10, 8))
    axs[1].plot(emrad_data.rad1.I)
    axs[1].plot(emrad_data.rad1.Q)
    axs[2].plot(emrad_data.rad2.I)
    axs[2].plot(emrad_data.rad2.Q)
    axs[3].plot(emrad_data.rad3.I)
    axs[3].plot(emrad_data.rad3.Q)
    axs[4].plot(emrad_data.rad4.I)
    axs[4].plot(emrad_data.rad4.Q)

    for _, row in log.iterrows():
        axs[0].axvspan(row.start_time, row.end_time, color="red", alpha=0.3)
        axs[1].axvspan(row.start_time, row.end_time, color="red", alpha=0.3)
        axs[2].axvspan(row.start_time, row.end_time, color="red", alpha=0.3)
        axs[3].axvspan(row.start_time, row.end_time, color="red", alpha=0.3)
        axs[4].axvspan(row.start_time, row.end_time, color="red", alpha=0.3)

    # write row.activity as text in the middle of the span
    for _, row in log.iterrows():
        mid_time = row.start_time + (row.end_time - row.start_time) / 2
        axs[0].text(mid_time, 0, row.pattern, color="black", fontsize=8, ha="center", va="bottom", rotation=90)

    axs[0].set_title(f"{subject} - Activities Overview")
    axs[0].set_ylabel("Activities")
    axs[1].set_ylabel("Radar 1")
    axs[2].set_ylabel("Radar 2")
    axs[3].set_ylabel("Radar 3")
    axs[4].set_ylabel("Radar 4")


def sync_raw_plot(emrad_data: pd.DataFrame, biopac_data: pd.DataFrame, psg_data: pd.DataFrame, subject: str):
    fig, axs = plt.subplots(3, sharex=True)
    axs[0].plot(biopac_data["sync"])
    axs[1].plot(emrad_data["rad1"]["Sync_In"])
    axs[2].plot(psg_data["sync"])

    axs[0].set_title(f"{subject} - Sync Signals")
    axs[0].set_ylabel("BIOPAC Sync")
    axs[1].set_ylabel("EMRAD Sync")
    axs[2].set_ylabel("PSG Sync")


def sync_synced_plot(data: pd.DataFrame, subject: str):

    fs = 1000
    window_samples = 10 * fs

    # left column: start of the recording, right column: end of the recording
    slices = ((0, slice(None, window_samples)), (1, slice(-window_samples, None)))

    fig, axs = plt.subplots(3, 2, sharex='col')
    for col, data_slice in slices:
        axs[0, col].plot(data["biopac"]["sync"].iloc[data_slice])
        axs[1, col].plot(data["rad1"]["Sync_In"].iloc[data_slice])
        axs[2, col].plot(data["psg"]["sync"].iloc[data_slice])

    axs[0, 0].set_title(f"Start")
    axs[0, 1].set_title(f"End")
    axs[0, 0].set_ylabel("BIOPAC Sync")
    axs[1, 0].set_ylabel("EMRAD Sync")
    axs[2, 0].set_ylabel("PSG Sync")

    # only three x-ticks per column: start, middle, and end of the plotted window. The subplots share the x-axis
    # per column, so setting the ticks on the bottom axis is sufficient.
    for col, data_slice in slices:
        index = data["rad1"]["Sync_In"].iloc[data_slice].index
        if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
            # show the tick labels in local time
            index = index.tz_convert(TIMEZONE)
            axs[2, col].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=TIMEZONE))
        elif isinstance(index, pd.DatetimeIndex):
            # a timezone-naive index is plotted as-is, so the labels must not be converted either
            axs[2, col].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        axs[2, col].set_xticks([index[0], index[len(index) // 2], index[-1]])

    fig.autofmt_xdate()

    fig.suptitle(f"{subject} - Synced Signals")

def respiration_overview_synced_plotting(data: pd.DataFrame, log: pd.DataFrame, subject: str):
    """Plot all synchronized data channels with the breathing patterns as overlay.

    The sync channels are not plotted. For each radar, the I and Q channels are plotted in the same subplot,
    with their mean subtracted so that both channels are on a comparable scale.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        Synchronized data with a :class:`~pandas.MultiIndex` column level "type" (device) and the channels
        as second level.
    log : :class:`~pandas.DataFrame`
        Timelog with the columns "start_time", "end_time", and "pattern".
    subject : str
        Subject id, used for the figure title.

    """
    # collect the subplots: one per non-radar channel, one per radar (I and Q together)
    panels = []
    for device in data.columns.get_level_values(0).unique():
        data_device = data[device]
        channels = [c for c in data_device.columns if c.lower() not in SYNC_CHANNELS]

        if device.startswith("rad"):
            # plot I and Q of one radar in the same subplot, mean-reduced
            panels.append((device, [(c, data_device[c] - data_device[c].mean()) for c in channels]))
        else:
            panels.extend((f"{device}\n{c}", [(c, data_device[c])]) for c in channels)

    fig, axs = plt.subplots(len(panels), 1, sharex=True, figsize=(12, 1.6 * len(panels)))
    axs = np.atleast_1d(axs)

    for ax, (label, channels) in zip(axs, panels):
        for channel_name, channel_data in channels:
            ax.plot(channel_data, label=channel_name, linewidth=0.8)
        ax.set_ylabel(label)
        if len(channels) > 1:
            ax.legend(loc="upper right", fontsize=7)

        # overlay the breathing patterns
        for _, row in log.iterrows():
            ax.axvspan(row.start_time, row.end_time, color="red", alpha=0.3)

    # write the pattern of each phase in the middle of the span, above the first subplot
    for _, row in log.iterrows():
        mid_time = row.start_time + (row.end_time - row.start_time) / 2
        axs[0].text(
            mid_time, 1.02, row.pattern, color="black", fontsize=8, ha="center", va="bottom", rotation=90,
            transform=axs[0].get_xaxis_transform(),
        )

    axs[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[-1].xaxis.get_major_locator(), tz=TIMEZONE))
    fig.suptitle(f"{subject} - Synced Data Overview")
    fig.tight_layout()

    return fig, axs
