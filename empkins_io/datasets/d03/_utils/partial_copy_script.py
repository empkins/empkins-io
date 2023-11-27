from shutil import copy
import click
from pathlib import Path

""" This is a script to copy selected parts of the study data.

You can either run it without any arguments and just answer all the prompts, 
or you can run it completely non-interactive via the command line by setting all the flags.

For more information run `python selective_copy.py -h` or `python selective_copy.py --help`.

Note: This script has only been tested on the Micro dataset.
"""

@click.command()
@click.option(
    "-i",
    "--source",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to the data_per_subject folder",
    prompt="Specify the source folder (data_per_subject)",
)
@click.option(
    "-o",
    "--destination",
    type=click.Path(exists=True, file_okay=False, path_type=Path, writable=True),
    help="Path to the destination folder",
    prompt="Specify the destination folder",
)
@click.option(
    "-s",
    "--subjects",
    type=str,
    help="List of subjects to copy separated by spaces (e.g. 1 2 7 15 45), default is all",
    prompt="Specify the subjects to copy seperated by spaces (e.g. 1 2 7 15 45)",
    default="all",
)
@click.option(
    "-c",
    "--condition",
    type=click.Choice(["tsst", "ftsst", "both"], case_sensitive=False),
    default="both",
    help="Condition to copy, default is both.",
    prompt="Do you want to copy the TSST, FTSST or both?",
)
@click.option(
    "--include-videos/--exclude-videos",
    "-v/-nv",
    default=True,
    prompt="Do you want to include the video subfolder?",
    help="Include videos in the copy",
)
@click.option(
    "--include-emrad/--exclude-emrad",
    "-e/-ne",
    default=True,
    prompt="Do you want to include radar data (emrad)?",
    help="Include radar data (emrad) in the copy",
)
@click.option(
    "--include-nilspod/--exclude-nilspod",
    "-n/-nn",
    default=True,
    prompt="Do you want to include nilspod data?",
    help="Include nilspod data in the copy",
)
@click.option(
    "--include-biopac/--exclude-biopac",
    "-b/-nb",
    default=True,
    prompt="Do you want to include biopac data?",
    help="Include biopac data in the copy",
)
@click.option(
    "--include-timelogs/--exclude-timelogs",
    "-t/-nt",
    default=True,
    prompt="Do you want to include timelog data?",
    help="Include timelog data in the copy",
)
@click.option(
    "--dryrun",
    "-d",
    is_flag=True,
    default=False,
    help="Dryrun, only print the files that would be copied",
)
def partial_copy(
    source: Path,
    destination: Path,
    subjects: str,
    condition: str,
    include_videos: bool,
    include_emrad: bool,
    include_nilspod: bool,
    include_biopac: bool,
    include_timelogs: bool,
    dryrun: bool,
):
    """Copy certain data (chosen based on the specified conditions) from the source folder to the
    destination folder."""
    print("Source:", source)
    print("Destination:", destination)
    print("Subjects:", subjects)
    print("Condition:", condition)
    print("Include video:", include_videos)
    print("Include emrad:", include_emrad)
    print("Include nilspod:", include_nilspod)
    print("Include biopac:", include_biopac)
    print("Include timelogs:", include_timelogs)
    files = get_all_files_in_dir(source)
    blocklist = build_blocklist(
        condition,
        include_videos,
        include_emrad,
        include_nilspod,
        include_biopac,
        include_timelogs,
    )
    # exclude all files that are in folders from the blocklist
    wanted_files = [
        f for f in files if not any(b in str(f).split("/") for b in blocklist)
    ]
    if subjects != "all":
        subjects = [int(s) for s in subjects.split(" ")]
        allowlist = build_allowlist(subjects)
        # only include files that are in folders from the allowlist (subjects)
        wanted_files = [
            f for f in wanted_files if any(a in str(f).split("/") for a in allowlist)
        ]

    if dryrun:
        print("These files would be copied:")
    else:
        print("These files will be copied:")
    pretty_print_files(wanted_files)
    click.confirm("Do you want to copy these files now?", abort=True)
    copy_files(wanted_files, destination, dryrun)
    print()


def get_all_files_in_dir(source: Path):
    """Get all the files in the source folder."""
    files = source.glob("**/*")
    files = [file for file in files if file.is_file()]
    return files


def build_allowlist(subjects):
    """Build the list of allowed folders."""
    allowlist = []
    for subject in subjects:
        allowlist.append(f"VP_{subject:02d}")
    return allowlist


def build_blocklist(
    condition,
    include_videos,
    include_emrad,
    include_nilspod,
    include_biopac,
    include_timelogs,
):
    """Build the blocklist of files to exclude.
    This should be invoked on the filepath split by /, so essentially the
    entries in the blocklist are the folders that should be excluded."""
    blocklist = []
    if condition == "tsst":
        blocklist.append("ftsst")
    elif condition == "ftsst":
        blocklist.append("tsst")

    if not include_videos:
        blocklist.append("video")
    if not include_emrad:
        blocklist.append("emrad")
    if not include_nilspod:
        blocklist.append("nilspod")
    if not include_biopac:
        blocklist.append("biopac")
    if not include_timelogs:
        blocklist.append("timelog")
    return blocklist


def copy_files(wanted_files, destination, dryrun):
    """Copy the files to the destination folder."""
    for file in wanted_files:
        destination_file = (
            destination / "data_per_subject" / str(file).split("data_per_subject/")[1]
        )
        print("Making parent dirs:", destination_file.parent)
        if not dryrun:
            destination_file.parent.mkdir(parents=True, exist_ok=True)
        print("Copying:", file, "to", destination_file)
        if not dryrun:
            copy(file, destination_file)


def pretty_print_files(files: list):
    """Pretty print the files."""
    for f in files:
        print(str(f).split("data_per_subject")[1])


if __name__ == "__main__":
    partial_copy()

# TODO: logfile output
