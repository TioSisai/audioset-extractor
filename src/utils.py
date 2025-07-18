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
LOG_ROOT = PROJECT_ROOT / "logs"
AUDIO_ROOT = PROJECT_ROOT / "audio"


# ============================== Module Initialization ==============================
def _initialize_directories() -> None:
    """Creates global directories required for the project."""
    for path in [PICKLE_ROOT, METADATA_ROOT, ZIP_ROOT, LOG_ROOT, AUDIO_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


_initialize_directories()


# ============================== Private Implementations ==============================
def _proc_single_archive(archive_path: str | Path, left_strip: int | None = None) -> dict[str, tuple[str, str]]:
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
#     target_dir.mkdir(parents=True, exist_ok=True)
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


def _extract_files(stem_to_archive_and_innerfile: dict[str, tuple[str, str]], stems: list[str], entry: str) -> None:
    """
    Batch extracts specified audio files from archives.
    This version is optimized to group extractions by archive to reduce process overhead.

    Args:
        stem_to_archive_and_innerfile: Dictionary mapping file stems to (archive path, inner file path) tuples.
        stems: List of stems of audio files to extract.
        entry: Entry name used to create the target subdirectory (e.g., "strong_train").

    Raises:
        KeyError: If a stem in `stems` does not exist in the mapping dictionary.
        FileNotFoundError: If the 7z command-line tool is not installed.
        subprocess.CalledProcessError: If the 7z extraction command fails.
    """
    logger.info("Starting audio file extraction...")

    # 1. Group files to be extracted by their source archive
    archive_to_innerfiles: dict[Path, list[str]] = defaultdict(list)
    for stem in stems:
        try:
            archive_path_in_root, innerfile = stem_to_archive_and_innerfile[stem]
            abs_archive_path = ZIP_ROOT / archive_path_in_root
            archive_to_innerfiles[abs_archive_path].append(innerfile)
        except KeyError:
            logger.warning(f"Stem '{stem}' not found in the mapping. Skipping.")
            continue

    # 2. Create target directory
    target_dir = AUDIO_ROOT / entry
    target_dir.mkdir(parents=True, exist_ok=True)

    # 3. Execute one 7z command per archive
    extracted_cnt = 0
    for abs_archive_path, innerfiles in archive_to_innerfiles.items():
        # The -o switch must not have a space between it and the path.
        output_dir_switch = f"-o{target_dir}"
        cmds = ["7z", "e", str(abs_archive_path), output_dir_switch] + innerfiles

        try:
            subprocess.run(cmds, check=True)
            extracted_cnt += len(innerfiles)
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Failed to extract files from {abs_archive_path.name}. Return code: {e.returncode}. Error: {e.stderr}"
            )
            raise
        except FileNotFoundError:
            logger.error("7z command not found, please ensure p7zip is installed.")
            raise FileNotFoundError("Please install the 7z command-line tool (p7zip).")

    logger.info(f"Successfully extracted {extracted_cnt} audio files.")

# ============================== Public API ==============================


def archive_files_under(zip_root: Path) -> list[Path]:
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


def download_url(url: str, dest: str | Path) -> None:
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


def download_all_metafiles():
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
            download_url(file_url, dest_path)
            logger.info(f"{file_name} download complete.")


def prepare_all_pickles():
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

        unbalanced_archive_files = archive_files_under(ZIP_ROOT / "unbalanced_train_segments")
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


def process_entry(entry: str) -> None:
    """
    Processes the specified dataset entry.

    This process includes:
    1. Ensuring all metadata files are downloaded.
    2. Preparing a mapping (pickle file) containing paths to all audio files within archives.
    3. Reading metadata for the specified entry to get the list of required audio file stems.
    4. Comparing the required list with the available list, logging and reporting missing files.
    5. Extracting all available audio files from archives to the specified directory.

    Args:
        entry: Name of the processing option, e.g., "weak_unbalanced_train", "weak_balanced_train", "weak_eval", "strong_train", "strong_eval".
    """
    assert entry in [
        "weak_unbalanced_train",
        "weak_balanced_train",
        "weak_eval",
        "strong_train",
        "strong_eval",
    ], f"Invalid processing option: {entry}"

    download_all_metafiles()
    stem_to_archive_and_innerfile = prepare_all_pickles()

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
    logger.info(f"In {metafile.name}:")
    logger.info(f"	Found {len(unique_stems)} audio files.")
    logger.info(f"	Downloaded {len(intersection_stems)} audio files.")
    logger.info(
        f"	Missing {len(unique_stems - intersection_stems)} audio files, missing rate: {len(unique_stems - intersection_stems) / len(unique_stems):.2%}"
    )
    with open(LOG_ROOT / f"{entry}_missing_files.log", "w") as f:
        for stem in unique_stems - intersection_stems:
            f.write(f"{stem}.wav\n")
    logger.info(f"Recorded missing audio files for {entry} to {LOG_ROOT / f'{entry}_missing_files.log'}")
    _extract_files(stem_to_archive_and_innerfile, list(intersection_stems), entry)
    logger.info(f"Audio files for {entry} have been extracted to {AUDIO_ROOT / entry}")
