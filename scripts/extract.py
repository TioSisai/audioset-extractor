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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audioset Extractor")
    parser.add_argument("--weak_unbalanced_train", action="store_true", default=False,
                        help="Process unbalanced train data with weak labels")
    parser.add_argument("--weak_balanced_train", action="store_true", default=False,
                        help="Process balanced train data with weak labels")
    parser.add_argument("--weak_eval", action="store_true", default=False, help="Process eval data with weak labels")
    parser.add_argument("--strong_train", action="store_true", default=False,
                        help="Process train data with strong labels")
    parser.add_argument("--strong_eval", action="store_true", default=False,
                        help="Process eval data with strong labels")
    args = parser.parse_args()
    if not any([args.weak_unbalanced_train, args.weak_balanced_train, args.weak_eval, args.strong_train, args.strong_eval]):
        logger.error("Please specify at least one processing option.")
        parser.print_help()
        exit(1)
    enabled_options = [key for key, value in vars(args).items() if value]
    logger.info(f"Activated option(s): {', '.join(enabled_options)}")

    for option in enabled_options:
        process_entry(option)
