import re
import glob
import pathlib
import asyncio

import aiofiles
import pandas as pd
import numpy as np

async def read_file(file_name):
    async with aiofiles.open(file_name, mode='rb') as f:
        return await f.read()


def read_files_async(file_names: list) -> list[bytes]:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        asyncio.gather(*[read_file(file_name) for file_name in file_names]))

def read_txts(dir_path: pathlib.Path | str):
    files = glob.glob(str(dir_path) + "/*.txt")
    files.sort()
    pd.DataFrame(columns = ["bins", "date"])
    
    contents = read_files_async(files)
        
    return files
