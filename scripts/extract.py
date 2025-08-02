"""
Extracts audio data from Audioset archive with various processing options.
"""

# ============================== Standard Library ==============================
import argparse
import logging


# ============================== Third-party Libraries ==============================
import rootutils


# ============================== Root Directory Configuration ==============================
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


# ============================== Local Modules ==============================
from src.utils import process_entry


# ============================== Module-level Logger ==============================
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audioset Extractor")
    parser.add_argument("-u", "--weak_unbalanced_train", action="store_true", default=False,
                        help="Process unbalanced train data with weak labels")
    parser.add_argument("-b", "--weak_balanced_train", action="store_true", default=False,
                        help="Process balanced train data with weak labels")
    parser.add_argument("-e", "--weak_eval", action="store_true", default=False,
                        help="Process eval data with weak labels")
    parser.add_argument("-st", "--strong_train", action="store_true", default=False,
                        help="Process train data with strong labels")
    parser.add_argument("-se", "--strong_eval", action="store_true", default=False,
                        help="Process eval data with strong labels")
    parser.add_argument("-n", "--num_workers", type=int, default=None,
                        help="Number of parallel workers for processing, defaults to all available cores")
    parser.add_argument("-m", "--min_duration", type=float, default=None,
                        help="Minimum duration of audio files to process, defaults to detect non-empty files")
    parser.add_argument("-s", "--silence_db_threshold", type=float, default=None,
                        help="Silence threshold in dB for audio processing, considering audiofile with max dB < this value as silence, defaults to deactivate silence detection")

    args = parser.parse_args()
    if not any([args.weak_unbalanced_train, args.weak_balanced_train, args.weak_eval, args.strong_train, args.strong_eval]):
        logger.error("Please specify at least one processing option.")
        parser.print_help()
        exit(1)
    # 只筛选五个处理选项
    entry_options = [
        "weak_unbalanced_train",
        "weak_balanced_train",
        "weak_eval",
        "strong_train",
        "strong_eval",
    ]
    enabled_entries = [opt for opt in entry_options if getattr(args, opt)]
    logger.info(f"Activated option(s): {', '.join(enabled_entries)}")

    for entry in enabled_entries:
        process_entry(entry, num_workers=args.num_workers,
                      min_duration=args.min_duration,
                      silence_db_threshold=args.silence_db_threshold)

    logger.info("All processing completed.")
