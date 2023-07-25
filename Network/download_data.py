from os import mkdir
from shutil import rmtree
from globals import DATASET_PATH, LICHESS_DATA_SET, LICHESS_DATASET_PATH

import tqdm
import requests
import zstandard
from io import TextIOWrapper

try:
    mkdir(DATASET_PATH)
except Exception:
    rmtree(DATASET_PATH)
    mkdir(DATASET_PATH)


def download_file(url: str, path: str, limit: int = 1024):
    r = requests.get(url, stream=True)

    with open(path, 'wb') as file:
        for i, block in tqdm.tqdm(enumerate(r.iter_content(chunk_size=1024 * 1024)), total=limit, unit='MB'):
            if block:
                file.write(block)
            if i > limit:
                break

def uncompress_file(source_path: str, target_path: str):
    with open(source_path, 'rb') as fh:
        dctx = zstandard.ZstdDecompressor(max_window_size=2147483648)
        stream_reader = dctx.stream_reader(fh)
        text_stream = TextIOWrapper(stream_reader, encoding='utf-8')
        with open(target_path, 'w', encoding='utf-8') as target_file:
            target_file.write(text_stream.read())

FILE_SIZE_IN_MB = 4

download_file(
    f'https://database.lichess.org/standard/{LICHESS_DATA_SET}.zst', 
    f'{DATASET_PATH}/{LICHESS_DATA_SET}.zst', 
    FILE_SIZE_IN_MB
)

uncompress_file(
    source_path=f'{DATASET_PATH}/{LICHESS_DATA_SET}.zst',
    target_path=LICHESS_DATASET_PATH
)
