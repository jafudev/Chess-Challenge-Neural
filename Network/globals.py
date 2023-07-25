import os

LICHESS_DATA_SET = "lichess_db_standard_rated_2023-05.pgn"
TARGET_PATH = "dataset"
PROCESSED_DATA_CSV_NAME = "processed.csv"

TRAINED_MODEL_PATH = "trained_model"
OUTPUT_CSHARP_MODEL = "csharpmodel.txt"
PLOTS_DIRECTORY = "plots/"


current_directory = os.path.dirname(os.path.abspath(__file__))


DATASET_PATH = f"{current_directory}/{TARGET_PATH}"
LICHESS_DATASET_PATH = f"{DATASET_PATH}/{LICHESS_DATA_SET}"
PROCESSED_DATA_CSV_PATH = f"{DATASET_PATH}/{PROCESSED_DATA_CSV_NAME}"

TRAINED_MODEL_PATH = f"{current_directory}/{TRAINED_MODEL_PATH}"
OUTPUT_CSHARP_MODEL_PATH = f"{TRAINED_MODEL_PATH}/{OUTPUT_CSHARP_MODEL}"
PLOTS_PATH = f"{TRAINED_MODEL_PATH}/{PLOTS_DIRECTORY}"

