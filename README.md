# Audioset Extractor

Since the AudioSet dataset is very large, some researchers may only want to use a small subset of it (e.g., weak-label balanced training set, weak-label evaluation set, strong-label training set, strong-label validation set, etc.). Therefore, this repository aims to provide a simpler script to extract specific subsets from the archive files provided by PANNs, making it easier for researchers to use this dataset for downstream tasks.

## Features

- Extract specific audio subsets from AudioSet archive files.
- Supports custom selection of subsets to extract, including:
    * Weak unbalanced train
    * Weak balanced train
    * Weak eval
    * Strong train
    * Strong eval
- Automatically downloads the required official metadata files.
- Efficiently handles large archive files by batch-extracting per archive to improve performance.
- Logs and reports audio files that are present in the metadata but missing from the archive files.

## Prerequisites

Before you begin, ensure you have the following dependencies installed:

1.  **Python 3.x**
2.  **p7zip**: For handling `.zip` archive files.
    -   On Debian/Ubuntu: `sudo apt-get install p7zip-full`
    -   On macOS (with Homebrew): `brew install p7zip`
    -   On Windows: Please download and install 7-Zip from the [official 7-Zip website](https://www.7-zip.org/) and add the path to the 7z executable to your system's environment variables.
3.  **wget** or **curl** (at least one must be available): For downloading metadata files.
4.  **Python Dependencies**: Run the following command in the project root directory to install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Directory Structure

Before running the script, ensure your directory structure is as follows:

```
audioset-extractor/
├── audio/                # (Auto-created) Stores the extracted .wav audio files
├── logs/                 # (Auto-created) Stores logs of missing files
├── metadata/             # (Auto-created) Stores downloaded Audioset metadata
├── pickles/              # (Auto-created) Stores pre-processed audio file indexes
├── scripts/
│   └── extract.py        # The main extraction script
├── src/
└── zip_audios/           # <--- Place the Audioset archive folder (zip_audios) downloaded from the PANNs-provided archives here
    ├── balanced_train_segments.zip
    ├── eval_segments.zip
    └── unbalanced_train_segments/
        ├── unbalanced_train_segments_part00.zip
        ├── unbalanced_train_segments_part01.zip
        └── ...
```

## Usage

1.  **Download Archive Files**
    Download the Audioset archive files from the [PANNs provided archival URL (Their Baidu Cloud disk link)](https://github.com/qiuqiangkong/audioset_tagging_cnn#1-download-dataset) and place its inner `zip_audios/` folder into the project root directory.

2.  **Run the Extraction Script**
    Open a terminal, use `scripts/extract.py`, and specify the datasets to extract via command-line arguments.

    **Examples:**

    -   Extract all available datasets:
        ```bash
        cd /path/to/audioset-extractor
        python scripts/extract.py --weak_unbalanced_train --weak_balanced_train --weak_eval --strong_train --strong_eval
        ```

    -   Extract only the strong-label training and evaluation sets:
        ```bash
        cd /path/to/audioset-extractor
        python scripts/extract.py --strong_train --strong_eval
        ```

    -   Extract only the weak-label balanced training set:
        ```bash
        cd /path/to/audioset-extractor
        python scripts/extract.py --weak_balanced_train
        ```

3.  **Check the Output**
    -   The extracted audio files will be located in the `audio/` directory, in subdirectories corresponding to the arguments you specified (e.g., `strong_train`).
    -   Since many videos on YouTube have been removed, we cannot do anything about this part. For audio files that are listed in the metadata but not found in the archives, a list of missing files will be saved in the `logs/` directory.

## Acknowledgements

The Audioset archive files used in this project are provided by the [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) project. Special thanks to them.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
