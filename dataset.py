import os
import pandas as pd
import zstandard as zstd

input_path = "00.jsonl.zst"
output_path = "00.output.jsonl"

dctx = zstd.ZstdDecompressor()
with open(input_path, 'rb') as ifh, open(output_path, 'wb') as ofh:
    dctx.copy_stream(ifh, ofh, read_size=1000, write_size=1000)