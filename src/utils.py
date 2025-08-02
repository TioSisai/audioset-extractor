"""
Provides utility functions for handling files and archives.
"""

# ============================== Standard Library ==============================
import logging
import pickle
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
import json
import re
import os
from multiprocessing import Pool, cpu_count
from typing import Union, Dict, Tuple
from functools import partial

# ============================== Third-party Libraries ==============================
import pandas as pd
import rootutils
import tqdm

# ============================== Module-level Logger ==============================
logger = logging.getLogger(__name__)

# ============================== Module-level Constants ==============================
PROJECT_ROOT = rootutils.find_root(__file__, indicator=".project-root")
PICKLE_ROOT = PROJECT_ROOT / "pickles"
METADATA_ROOT = PROJECT_ROOT / "metadata"
ZIP_ROOT = PROJECT_ROOT / "zip_audios"
MISSING_ROOT = PROJECT_ROOT / "missing_files"
AUDIO_ROOT = PROJECT_ROOT / "audio"
INVALID_ROOT = PROJECT_ROOT / "invalid_audio"


# ============================== Private Implementations ==============================
def _initialize_directories(entry: str) -> None:
    """Creates global directories required for the project."""
    for path in [PICKLE_ROOT, METADATA_ROOT, ZIP_ROOT, MISSING_ROOT, AUDIO_ROOT, INVALID_ROOT]:
        path.mkdir(parents=True, exist_ok=True)
    (AUDIO_ROOT / entry).mkdir(parents=True, exist_ok=True)
    (INVALID_ROOT / entry).mkdir(parents=True, exist_ok=True)


def _proc_single_archive(archive_path: str | Path, left_strip: int = 1) -> dict[str, tuple[str, str]]:
    """
    Lists .wav files within an archive using p7zip.

    Args:
        archive_path: Path to the archive file (or the first part if it's a multi-volume archive).
        left_strip: Number of characters to strip from the left of the filename stem.

    Returns:
        A dictionary mapping file stems to (archive path, inner file path) tuples.

    Raises:
        FileNotFoundError: If the 7z command or the archive file does not exist.
        subprocess.CalledProcessError: If the 7z command execution fails.
        ValueError: If the 7z output cannot be parsed.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"The specified archive file does not exist: {archive_path}")

    # 1. Construct command list
    command = ["7z", "l", str(archive_path)]

    try:
        # 2. Execute command
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # 3. Parse output to extract filenames
        output_lines = result.stdout.splitlines()
        stem_to_archive_and_innerfile: dict[str, tuple[str, str]] = {}

        with tqdm.tqdm(total=len(output_lines), desc=f"Processing {archive_path.name}", unit="line") as pbar:
            for line in output_lines:
                if line.endswith(".wav"):
                    # 7z output is fixed-width, filename is in the last part
                    innerfile = line.split()[-1]
                    # Core logic: extract filename stem from the inner file path.
                    # Example: "folder/audio_01.wav" -> "audio_01"
                    # left_strip is used to remove specific characters from the beginning of the stem.
                    stem_key = (
                        innerfile.split("/")[-1].rsplit(".", maxsplit=1)[0][left_strip:]
                        if left_strip
                        else innerfile.split("/")[-1].rsplit(".", maxsplit=1)[0]
                    )
                    stem_to_archive_and_innerfile[stem_key] = (str(archive_path.relative_to(ZIP_ROOT)), innerfile)
                pbar.update(1)

        return stem_to_archive_and_innerfile

    except FileNotFoundError as e:
        logger.error(f"7z command or archive file does not exist: {e}")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"7z command execution failed, return code: {e.returncode}, error output: {e.stderr}")
        raise
    except ValueError as e:
        logger.error(f"Failed to parse 7z output: {e}")
        raise

# # 保留这个函数的原始实现，可能会在未来回档
# def _extract_files(stem_to_archive_and_innerfile: dict[str, tuple[str, str]], stems: list[str], entry: str) -> None:
#     """
#     Batch extracts specified audio files from archives.

#     Args:
#         stem_to_archive_and_innerfile: Dictionary mapping file stems to (archive path, inner file path) tuples.
#         stems: List of stems of audio files to extract.
#         entry: Entry name used to create the target subdirectory (e.g., "strong_train").

#     Raises:
#         KeyError: If a stem in `stems` does not exist in the mapping dictionary.
#         FileNotFoundError: If the 7z command-line tool is not installed.
#         subprocess.CalledProcessError: If the 7z extraction command fails.
#     """
#     logger.info(f"Starting audio file extraction...")
#     extracted_cnt = 0
#     target_dir = AUDIO_ROOT / entry
#     target_dir = str(target_dir)
#     with tqdm.tqdm(total=len(stems), desc="Extracting audio files", unit="file") as pbar:
#         for stem in stems:
#             archive_path_in_root, innerfile = stem_to_archive_and_innerfile[stem]
#             abs_archive_path = ZIP_ROOT / archive_path_in_root
#             cmds = ["7z", "e", str(abs_archive_path), "-o" + target_dir, innerfile]
#             try:
#                 subprocess.run(cmds, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#                 extracted_cnt += 1
#             except subprocess.CalledProcessError as e:
#                 logger.error(f"Failed to extract {stem}, return code: {e.returncode}, error output: {e.stderr}")
#                 raise
#             except FileNotFoundError:
#                 logger.error(f"7z command not found, please ensure 7z is installed.")
#                 raise FileNotFoundError("Please install the 7z command-line tool to extract audio files.")
#             pbar.update(1)
#     logger.info(f"Successfully extracted {extracted_cnt} audio files.")


# def _extract_files(stem_to_archive_and_innerfile: dict[str, tuple[str, str]], stems: list[str], entry: str) -> None:
#     """
#     Batch extracts specified audio files from archives.
#     This version is optimized to group extractions by archive to reduce process overhead.

#     Args:
#         stem_to_archive_and_innerfile: Dictionary mapping file stems to (archive path, inner file path) tuples.
#         stems: List of stems of audio files to extract.
#         entry: Entry name used to create the target subdirectory (e.g., "strong_train").

#     Raises:
#         KeyError: If a stem in `stems` does not exist in the mapping dictionary.
#         FileNotFoundError: If the 7z command-line tool is not installed.
#         subprocess.CalledProcessError: If the 7z extraction command fails.
#     """
#     logger.info("Starting audio file extraction...")

#     # 1. Group files to be extracted by their source archive
#     archive_to_innerfiles: dict[Path, list[str]] = defaultdict(list)
#     for stem in stems:
#         try:
#             archive_path_in_root, innerfile = stem_to_archive_and_innerfile[stem]
#             abs_archive_path = ZIP_ROOT / archive_path_in_root
#             archive_to_innerfiles[abs_archive_path].append(innerfile)
#         except KeyError:
#             logger.warning(f"Stem '{stem}' not found in the mapping. Skipping.")
#             continue

#     # 2. Create target directory
#     target_dir = AUDIO_ROOT / entry

#     # 3. Execute one 7z command per archive
#     extracted_cnt = 0
#     for abs_archive_path, innerfiles in archive_to_innerfiles.items():
#         # The -o switch must not have a space between it and the path.
#         output_dir_switch = f"-o{target_dir}"
#         cmds = ["7z", "e", str(abs_archive_path), output_dir_switch] + innerfiles

#         try:
#             subprocess.run(cmds, check=True)
#             extracted_cnt += len(innerfiles)
#         except subprocess.CalledProcessError as e:
#             logger.error(
#                 f"Failed to extract files from {abs_archive_path.name}. Return code: {e.returncode}. Error: {e.stderr}"
#             )
#             raise
#         except FileNotFoundError:
#             logger.error("7z command not found, please ensure p7zip is installed.")
#             raise FileNotFoundError("Please install the 7z command-line tool (p7zip).")

#     logger.info(f"Successfully extracted {extracted_cnt} audio files.")


def _extract_worker(abs_archive_path: Path, innerfiles: list[str], target_dir: Path) -> tuple[int, str | None]:
    """
    Worker Function: Extracts files from a single archive.
    This function is designed to be called by multiprocessing.Pool.

    Args:
        abs_archive_path (Path): The absolute path to the archive.
        innerfiles (list[str]): A list of files to be extracted from this archive.
        target_dir (Path): The target directory for file extraction.

    Returns:
        A tuple containing (number of successfully extracted files, error message or None).
    """
    # The -y parameter automatically answers "yes" to all 7z prompts, preventing the process from hanging.
    output_dir_switch = f"-o{target_dir}"
    cmds = ["7z", "e", str(abs_archive_path), output_dir_switch, "-y"] + innerfiles

    try:
        # Using capture_output=True captures stdout and stderr for easier debugging.
        subprocess.run(cmds, check=True, capture_output=True, text=True, encoding='utf-8')
        return (len(innerfiles), None)
    except FileNotFoundError:
        # This error is fatal and the same for all worker processes.
        return (0, "Error: 7z command not found. Please ensure p7zip (or 7-Zip) is installed and in the system PATH.")
    except subprocess.CalledProcessError as e:
        # Catch extraction failure errors and return detailed information.
        error_msg = (
            f"Extraction from {abs_archive_path.name} failed. "
            f"Return code: {e.returncode}. Stderr: {e.stderr.strip()}"
        )
        return (0, error_msg)


def _extract_files(
    stem_to_archive_and_innerfile: dict[str, tuple[str, str]],
    stems: list[str],
    entry: str,
    num_processes: int | None = None
) -> None:
    """
    Batch extracts specified audio files from archives in parallel using multiple processes.

    Args:
        stem_to_archive_and_innerfile: A dictionary mapping file stems to (archive path, inner file path) tuples.
        stems: List of stems of audio files to extract.
        entry: Entry name used to create the target subdirectory (e.g., "strong_train").
        num_processes: Number of processes to use. If None, defaults to os.cpu_count().

    Raises:
        # No longer actively throws exceptions, but logs errors to enhance robustness.
    """
    # Use os.cpu_count() to get the number of CPU cores as the default number of processes.
    effective_processes = num_processes or os.cpu_count()
    logger.info(f"Starting audio file extraction... Will use {effective_processes} processes.")

    # 1. Group files to be extracted by their source archive (logic unchanged).
    archive_to_innerfiles: dict[Path, list[str]] = defaultdict(list)
    for stem in stems:
        try:
            archive_path_in_root, innerfile = stem_to_archive_and_innerfile[stem]
            abs_archive_path = ZIP_ROOT / archive_path_in_root
            archive_to_innerfiles[abs_archive_path].append(innerfile)
        except KeyError:
            logger.warning(f"Stem '{stem}' not found in the mapping, skipping.")
            continue

    if not archive_to_innerfiles:
        logger.info("No files to extract.")
        return

    # 2. Create target directory (logic unchanged).
    target_dir = AUDIO_ROOT / entry

    # 3. Prepare task list for the multiprocessing pool.
    # Each task is a tuple containing (archive path, file list, target directory).
    tasks = [
        (abs_archive_path, innerfiles, target_dir)
        for abs_archive_path, innerfiles in archive_to_innerfiles.items()
    ]

    # 4. Execute extraction tasks in parallel using a process pool.
    total_extracted_cnt = 0
    failed_archives_cnt = 0

    # Use a with statement to ensure the process pool is properly closed.
    with Pool(processes=num_processes) as pool:
        # pool.starmap unpacks each tuple in the tasks list as arguments to _extract_worker.
        results = pool.starmap(_extract_worker, tasks)

    # 5. Process and summarize the return results of all processes.
    for count, error_msg in results:
        if error_msg:
            logger.error(error_msg)
            failed_archives_cnt += 1
        else:
            total_extracted_cnt += count

    # 6. Final log summary.
    logger.info(f"Successfully extracted {total_extracted_cnt} audio files.")
    if failed_archives_cnt > 0:
        logger.warning(
            f"There were {failed_archives_cnt} archives that failed to extract. Please check the error logs above.")


def _rename_audios_in_directory(directory: Path) -> None:
    for audio_file in directory.glob("*.wav"):
        orig_name = audio_file.name
        # Remove the 'Y' prefix from the archive file provided by PANNs repository, which is not as the same as Google's Documentation.
        new_name = orig_name[1:]
        new_path = audio_file.parent / new_name
        audio_file.rename(new_path)


def _check_audio_file_worker(
    audio_file: Path,
    min_duration: float = None,
    silence_db_threshold: float = None
) -> Union[Tuple[str, str], None]:
    """
    Worker function to check a single audio file.

    Returns:
        A tuple (filename, reason_string) if any check fails, otherwise None.
    """
    # Check if the file exists
    if not audio_file.exists():
        return (audio_file.name, "File not found")

    # --- Step 1: Metadata validation using ffprobe ---
    try:
        command_info = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(audio_file)
        ]
        result_info = subprocess.run(
            command_info, capture_output=True, text=True, check=True, encoding="utf-8"
        )
        data = json.loads(result_info.stdout)

        # Check 2: Duration
        duration = float(data.get("format", {}).get("duration", "0"))
        if min_duration is not None:
            if duration < min_duration:
                reason = f"Duration too short or emptly file: {duration:.2f}s"
                logger.debug(f"'{audio_file.name}' failed duration check. Reason: {reason}")
                return (audio_file.name, reason)
        else:
            if duration == 0:
                return (audio_file.name, "File is empty")

        # Check 3 (Strict): Validate container and codec
        format_name = data.get("format", {}).get("format_name", "")
        audio_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
        codec_name = audio_stream.get("codec_name", "") if audio_stream else ""

        if "wav" not in format_name or "pcm" not in codec_name:
            reason = f"Invalid format (container: '{format_name}', codec: '{codec_name}')"
            logger.debug(f"'{audio_file.name}' failed format check. Reason: {reason}")
            return (audio_file.name, reason)

    except subprocess.CalledProcessError as e:
        reason = f"Corrupted or unreadable file. Stderr: {e.stderr.strip()}"
        logger.warning(f"ffprobe could not read '{audio_file.name}'. Reason: {reason}")
        return (audio_file.name, reason)
    except (json.JSONDecodeError, KeyError, ValueError):
        reason = "Failed to parse ffprobe output"
        logger.warning(f"Could not parse ffprobe output for '{audio_file.name}'.")
        return (audio_file.name, reason)
    except FileNotFoundError:
        logger.error("ffprobe command not found. Please ensure ffmpeg is installed in your PATH.")
        raise

    # --- Step 2: Silence detection using ffmpeg's volumedetect ---
    if silence_db_threshold is not None:
        try:
            command_volume = [
                "ffmpeg", "-i", str(audio_file), "-af", "volumedetect",
                "-vn", "-f", "null", "-",
            ]
            result_volume = subprocess.run(
                command_volume, capture_output=True, text=True, encoding="utf-8"
            )
            output = result_volume.stderr

            max_volume_match = re.search(r"max_volume:\s*(-?[\d\.]+) dB", output)
            if not max_volume_match:
                reason = "Could not determine max_volume"
                logger.warning(f"'{audio_file.name}' failed volume detection. Reason: {reason}")
                return (audio_file.name, reason)

            max_volume = float(max_volume_match.group(1))
            if max_volume < silence_db_threshold:
                reason = f"Silent file: max_volume is {max_volume:.1f} dB"
                logger.debug(f"'{audio_file.name}' failed silence check. Reason: {reason}")
                return (audio_file.name, reason)

        except FileNotFoundError:
            logger.error("ffmpeg command not found. Please ensure ffmpeg is installed in your PATH.")
            raise

    # All checks passed
    return None


def _archive_files_under(zip_root: Path) -> list[Path]:
    """
    Gets all archive files under the specified directory.

    Args:
        zip_root: Path to the directory containing archive files.

    Returns:
        A list containing paths to all archive files.
    """
    if not zip_root.exists() or not zip_root.is_dir():
        raise FileNotFoundError(f"The specified directory does not exist or is not a directory: {zip_root}")

    return list(zip_root.glob("*.zip"))


def _download_url(url: str, dest: str | Path) -> None:
    """
    Downloads a file from the specified URL to the target path.

    Args:
        url: URL of the file to download.
        dest: Target path to save the downloaded file.

    Raises:
        subprocess.CalledProcessError: If the download command execution fails.
    """
    # Check if wget or curl command is available locally
    if not shutil.which("wget") and not shutil.which("curl"):
        logger.error("Please install wget or curl command-line tools to download files.")
        raise FileNotFoundError("Please install wget or curl command-line tools to download files.")
    elif shutil.which("wget"):
        avail_dl = "wget"
    else:
        avail_dl = "curl"

    if isinstance(dest, Path):
        dest = str(dest)

    if avail_dl == "wget":
        command = ["wget", url, "-O", dest]
    else:  # curl
        command = ["curl", "-L", url, "-o", dest]

    try:
        logger.info(f"Starting download of {url} to {dest}...")
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info(f"Successfully downloaded {url} to {dest}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed, return code: {e.returncode}, error output: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error(f"{avail_dl} command not found, please ensure {avail_dl} is installed.")
        raise FileNotFoundError(f"Please install the {avail_dl} command-line tool to download files.")


def _download_all_metafiles():
    """Downloads all required metadata files."""
    download_urls: list[str] = [
        "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv",
        "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv",
        "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv",
        "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
        "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/qa/qa_true_counts.csv",
        "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/qa/rerated_video_ids.txt",
        "http://storage.googleapis.com/us_audioset/youtube_corpus/strong/audioset_train_strong.tsv",
        "http://storage.googleapis.com/us_audioset/youtube_corpus/strong/audioset_eval_strong.tsv",
        "http://storage.googleapis.com/us_audioset/youtube_corpus/strong/audioset_eval_strong_framed_posneg.tsv",
        "http://storage.googleapis.com/us_audioset/youtube_corpus/strong/mid_to_display_name.tsv",
    ]
    if not METADATA_ROOT.exists():
        METADATA_ROOT.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created metadata directory: {METADATA_ROOT}")
    for file_url in download_urls:
        file_name = file_url.split("/")[-1]
        dest_path = METADATA_ROOT / file_name
        if not dest_path.exists():
            logger.info(f"Starting download of {file_name} to {dest_path}...")
            _download_url(file_url, dest_path)
            logger.info(f"{file_name} download complete.")


def _prepare_all_pickles():
    """Prepares all required pickle files, mainly mappings of audio file paths."""
    PICKLE_ROOT.mkdir(parents=True, exist_ok=True)
    stem_to_archive_and_innerfile: dict[str, tuple[str, str]] = {}
    stem_to_archive_and_innerfile_pickle = PICKLE_ROOT / "stem_to_archive_and_innerfile.pkl"
    if stem_to_archive_and_innerfile_pickle.exists():
        with open(stem_to_archive_and_innerfile_pickle, "rb") as f:
            stem_to_archive_and_innerfile = pickle.load(f)
        logger.info(f"Loaded pickle file from {stem_to_archive_and_innerfile_pickle}.")
    else:
        logger.info("stem_to_archive_and_innerfile.pkl does not exist, starting processing.")

        unbalanced_archive_files = _archive_files_under(ZIP_ROOT / "unbalanced_train_segments")
        unbalanced_stem_to_archive_and_innerfile: dict[str, tuple[str, str]] = {}
        for unbalanced_archive_file in unbalanced_archive_files:
            unbalanced_stem_to_archive_and_innerfile.update(_proc_single_archive(unbalanced_archive_file, left_strip=1))
        logger.info(
            f"Processed {len(unbalanced_stem_to_archive_and_innerfile)} audio files for unbalanced training set.")
        stem_to_archive_and_innerfile.update(unbalanced_stem_to_archive_and_innerfile)

        balanced_archive_file = ZIP_ROOT / "balanced_train_segments.zip"
        balanced_stem_to_archive_and_innerfile = _proc_single_archive(balanced_archive_file, left_strip=1)
        logger.info(f"Processed {len(balanced_stem_to_archive_and_innerfile)} audio files for balanced training set.")
        stem_to_archive_and_innerfile.update(balanced_stem_to_archive_and_innerfile)

        eval_archive_file = ZIP_ROOT / "eval_segments.zip"
        eval_stem_to_archive_and_innerfile = _proc_single_archive(eval_archive_file, left_strip=1)
        logger.info(f"Processed {len(eval_stem_to_archive_and_innerfile)} audio files for evaluation set.")
        stem_to_archive_and_innerfile.update(eval_stem_to_archive_and_innerfile)

        with open(stem_to_archive_and_innerfile_pickle, "wb") as f:
            pickle.dump(stem_to_archive_and_innerfile, f)

        assert len(set(stem_to_archive_and_innerfile.keys())) == len(
            list(stem_to_archive_and_innerfile.keys())
        ), "Duplicate keys found in stem_to_archive_and_innerfile, please check the processing logic."
        logger.info(f"Processing complete, found {len(stem_to_archive_and_innerfile)} audio file mappings.")
        logger.info(f"Saved to {stem_to_archive_and_innerfile_pickle}")
        logger.info("All audio file pickle files are ready.")
    return stem_to_archive_and_innerfile


def _invalid_directory(
    directory_path: Union[str, Path],
    num_workers: int = None,
    min_duration: float = None,
    silence_db_threshold: float = None
) -> Dict[str, str]:
    """
    Checks all .wav files in a directory in parallel.

    Args:
        directory_path: The path to the directory containing .wav files.
        num_workers: The number of processes to use. Defaults to the number of CPU cores.
        min_duration: Minimum required duration in seconds.
        silence_db_threshold: The noise tolerance in dB for silence detection.

    Returns:
        A dictionary where keys are failed filenames and values are the reasons for failure.
    """
    directory_path = Path(directory_path)
    if not directory_path.is_dir():
        logger.error(f"Provided path is not a directory: {directory_path}")
        return {}

    files_to_check = list(directory_path.glob("*.wav"))

    if not files_to_check:
        logger.info(f"No .wav files found in directory: {directory_path}")
        return {}

    # If num_workers is not specified, use all available CPU cores
    workers = num_workers or cpu_count()
    logger.info(f"Found {len(files_to_check)} .wav files. Starting validation with {workers} workers...")

    worker_func = partial(
        _check_audio_file_worker,
        min_duration=min_duration,
        silence_db_threshold=silence_db_threshold
    )

    failed_files_map = {}
    with Pool(processes=workers) as pool:
        results_iterator = pool.imap_unordered(worker_func, files_to_check)

        for result in tqdm.tqdm(results_iterator, total=len(files_to_check), desc="Validating files"):
            if result is not None:
                filename, reason = result
                failed_files_map[filename] = reason

    logger.info(f"Validation complete. Found {len(failed_files_map)} problematic files.")
    for item_filename, item_reason in failed_files_map.items():
        logger.info(f"{item_filename}: {item_reason}")
    return failed_files_map


# ============================== Public API ==============================

def process_entry(entry: str, num_workers: int = None, min_duration: float = None, silence_db_threshold: float = None):
    """
    Processes a specified dataset entry from start to finish.

    This comprehensive process involves the following steps:
    1.  Ensures all required metadata files (e.g., CSVs, TSVs) are downloaded from their sources.
    2.  Prepares a mapping from audio file stems to their location within archive files.
        This mapping is cached in a pickle file for efficiency.
    3.  Reads the metadata corresponding to the specified `entry` to determine the list of required audio files.
    4.  Extracts the required and available audio files from their respective archives into a dedicated directory.
    5.  Validates the extracted audio files, checking for corruption, incorrect formats, minimum duration, and silence.
        Problematic files are identified and subsequently deleted.
    6.  Calculates and logs a summary, detailing the number of required, found, and missing audio files.
    7.  Saves a list of missing file stems to a text file for later review.
    8.  Performs a final renaming operation on the extracted audio files to standardize their names.

    Args:
        entry: The name of the dataset partition to process. Valid options include:
               "weak_unbalanced_train", "weak_balanced_train", "weak_eval",
               "strong_train", "strong_eval".
        num_workers: Number of worker processes to use for parallel processing. Defaults to the number of CPU cores.
        min_duration: Minimum required duration for audio files in seconds. If None, check only for empty files.
        silence_db_threshold: The noise tolerance in dB for silence detection. If None, no silence check is performed.
    """
    assert entry in [
        "weak_unbalanced_train",
        "weak_balanced_train",
        "weak_eval",
        "strong_train",
        "strong_eval",
    ], f"Invalid processing option: {entry}"

    _initialize_directories(entry)

    _download_all_metafiles()

    stem_to_archive_and_innerfile = _prepare_all_pickles()

    entry_meta_map: dict[str, Path] = {
        "weak_unbalanced_train": METADATA_ROOT / "unbalanced_train_segments.csv",
        "weak_balanced_train": METADATA_ROOT / "balanced_train_segments.csv",
        "weak_eval": METADATA_ROOT / "eval_segments.csv",
        "strong_train": METADATA_ROOT / "audioset_train_strong.tsv",
        "strong_eval": METADATA_ROOT / "audioset_eval_strong.tsv",
    }

    metafile = entry_meta_map.get(entry)

    if "strong_" in entry:
        df = pd.read_csv(metafile, sep="\t", engine="python")
        unique_filenames = df.iloc[:, 0].unique().tolist()
        unique_stems = set([item.rsplit("_", maxsplit=1)[0] for item in unique_filenames])
    else:
        df = pd.read_csv(metafile, header=2, sep=", ", engine="python")
        unique_filenames = df.iloc[:, 0].unique().tolist()
        unique_stems = set(unique_filenames)

    downloaded_stems = set(stem_to_archive_and_innerfile.keys())
    intersection_stems = unique_stems.intersection(downloaded_stems)
    logger.info(f"Starting extraction for {entry}...")
    _extract_files(stem_to_archive_and_innerfile, list(intersection_stems), entry)
    logger.info(f"Extraction for {entry} completed.")

    logger.info(f"Starting to check invalid or broken audio files for {entry}...")
    invalid_file_reason_map: dict = _invalid_directory(AUDIO_ROOT / entry, num_workers=num_workers,
                                                       min_duration=min_duration,
                                                       silence_db_threshold=silence_db_threshold)

    # Move invalid files to the INVALID_ROOT directory
    for invalid_file in invalid_file_reason_map.keys():
        invalid_file_path = AUDIO_ROOT / entry / invalid_file
        invalid_file_path.rename(INVALID_ROOT / entry / invalid_file[1:])

    invalid_stems = set([invalid_item[1:].rstrip(".wav") for invalid_item in list(invalid_file_reason_map.keys())])
    valid_and_existing_stems = intersection_stems - invalid_stems
    logger.info(f"Invalid file check for {entry} completed.")

    logger.info(f"In {metafile.name}:")
    logger.info(f"	Found {len(unique_stems)} audio files.")
    logger.info(f"	Downloaded {len(valid_and_existing_stems)} valid audio files.")
    logger.info(
        f"	Missing {len(unique_stems - valid_and_existing_stems)} audio files, missing rate: {len(unique_stems - valid_and_existing_stems) / len(unique_stems):.2%}"
    )

    with open(MISSING_ROOT / f"{entry}.txt", "w") as f:
        for stem in unique_stems - valid_and_existing_stems:
            f.write(f"{stem}.wav\n")
    logger.info(f"Recorded missing audio files for {entry} to {MISSING_ROOT / f'{entry}.txt'}")

    _rename_audios_in_directory(AUDIO_ROOT / entry)

    logger.info(f"Audio files for {entry} have been extracted to {AUDIO_ROOT / entry}")
