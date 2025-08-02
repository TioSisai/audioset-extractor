# Audioset Extractor

The Audioset dataset is vast, and some researchers may only need to use a small subset of it (e.g., the weak-label balanced training set, the strong-label validation set, etc.). This project aims to provide a simple and easy-to-use script to extract and validate specific subsets from the Audioset archives provided by the PANNs project, thereby simplifying the process for researchers to use this dataset in downstream tasks.

## Features

-   **Subset Extraction**: Extract specific audio subsets from Audioset archive files.
-   **Flexible Selection**: Supports custom selection of subsets to extract, including:
    -   Weak unbalanced train
    -   Weak balanced train
    -   Weak eval
    -   Strong train
    -   Strong eval
-   **Automatic Metadata Download**: Automatically downloads the required official metadata files.
-   **Efficient Parallel Processing**: Utilizes multiprocessing to extract and validate audio files in parallel, significantly improving efficiency when handling large batches of files.
-   **Comprehensive Audio Validation**:
    -   Uses `ffmpeg` to check for corrupted files or incorrect formats.
    -   Filters out audio files that are shorter than a specified duration threshold.
    -   Detects and filters out silent or excessively quiet audio files.
-   **Intelligent File Management**:
    -   Automatically moves files that fail validation (e.g., corrupted, silent) to a separate `invalid_audio/` directory for later review.
    -   Logs a list of audio files that are present in the metadata but missing from the archives, saving it to the `missing_files/` directory.

## Prerequisites

Before you begin, ensure you have the following dependencies installed:

1.  **Python 3.x**
2.  **p7zip**: For handling `.zip` archive files.
    -   On Debian/Ubuntu: `sudo apt-get install p7zip-full`
    -   On macOS (with Homebrew): `brew install p7zip`
    -   On Windows: Please download and install 7-Zip from the [official 7-Zip website](https://www.7-zip.org/) and add the path to the `7z.exe` executable to your system's environment variables.
3.  **ffmpeg**: For audio file validation (e.g., checking for corruption, getting duration, detecting silence).
    -   On Debian/Ubuntu: `sudo apt-get install ffmpeg`
    -   On macOS (with Homebrew): `brew install ffmpeg`
    -   On Windows: Please download it from the [official FFmpeg website](https://ffmpeg.org/download.html) and add it to your system's environment variables.
4.  **wget** or **curl** (at least one must be available): For downloading metadata files.
5.  **Python Dependencies**: Run the following command in the project root directory to install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Directory Structure

Before running the script, ensure your directory structure is as follows:

```
audioset-extractor/
├── audio/                # (Auto-created) Stores the extracted and valid .wav audio files
├── invalid_audio/        # (Auto-created) Stores audio files that failed validation (e.g., corrupted, silent)
├── missing_files/        # (Auto-created) Stores logs of missing files
├── metadata/             # (Auto-created) Stores downloaded Audioset metadata
├── pickles/              # (Auto-created) Stores pre-processed audio file indexes
├── scripts/
│   └── extract.py        # The main extraction script
├── src/
└── zip_audios/           # <--- Place the zip_audios folder downloaded from the PANNs archives here
    ├── balanced_train_segments.zip
    ├── eval_segments.zip
    └── unbalanced_train_segments/
        ├── unbalanced_train_segments_part00.zip
        ├── unbalanced_train_segments_part01.zip
        └── ...
```

## Usage

### 1. Download the Archive Files

Download the Audioset archive files from the [PANNs-provided archival URL](https://github.com/qiuqiangkong/audioset_tagging_cnn#1-download-dataset) and place its inner `zip_audios/` folder into the project root directory.

### 2. Run the Extraction Script

Open a terminal, use `scripts/extract.py`, and specify the datasets to extract via command-line arguments.

**Basic Examples:**

-   Extract all available datasets:
    ```bash
    python scripts/extract.py --weak_unbalanced_train --weak_balanced_train --weak_eval --strong_train --strong_eval
    ```

-   Extract only the strong-label training and evaluation sets:
    ```bash
    python scripts/extract.py --strong_train --strong_eval
    ```

-   Extract only the weak-label balanced training set:
    ```bash
    python scripts/extract.py --weak_balanced_train
    ```

**Advanced Usage (with Validation Parameters):**

You can use additional arguments to control the validation process:

-   **`--num_workers`**: Specifies the number of CPU cores to use for parallel processing (defaults to all available cores).
-   **`--min_duration`**: Sets the minimum valid duration for audio files in seconds. Files shorter than this will be considered invalid.
-   **`--silence_db_threshold`**: Sets the volume threshold (in dB) for silence detection. Files with a maximum volume below this threshold will be considered silent.

-   Example: Use 8 cores, and filter out files shorter than 1.5 seconds or with a max volume below -60 dB:
    ```bash
    python scripts/extract.py --strong_train --num_workers 8 --min_duration 1.5 --silence_db_threshold -60
    ```

### 3. Check the Output

-   **Valid Audio**: Extracted and validated audio files will be located in the `audio/` directory, organized into subdirectories corresponding to the arguments you specified (e.g., `strong_train`).
-   **Invalid Audio**: Files that failed validation (e.g., corrupted, too short, or silent) will be moved to the corresponding subdirectories in `invalid_audio/` for your review.
-   **Missing Files**: Since many videos on YouTube have been removed, we cannot do anything about this part. For audio files that are listed in the metadata but not found in the archives, a list of their filenames will be saved in the corresponding `.txt` file in the `missing_files/` directory.

## Acknowledgements

The Audioset archive files used in this project are provided by the [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) project. Special thanks to them.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
