#!/usr/bin/env python3
"""Creating an asyncio generator for blocks of audio data.

This example shows how a generator can be used to analyze audio input blocks.
In addition, it shows how a generator can be created that yields not only input
blocks but also output blocks where audio data can be written to.

You need Python 3.7 or newer to run this.

"""
import asyncio
import queue
import sys

import time

import numpy as np
import sounddevice as sd


async def inputstream_generator(channels=1, **kwargs):
    """Generator that yields blocks of input data as NumPy arrays."""
    q_in = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy()))

    stream = sd.InputStream(callback=callback, channels=channels, blocksize=256, **kwargs)
    with stream:
        while True:
            data = await q_in.get()
            yield data

async def block_stream_generator():
    BLOCK_LENGTH = 1024 # Must be multiple of 512
    block_number = int(BLOCK_LENGTH / 256)
    index_max = int(block_number - 1)
    index_split = int(block_number / 2)
    index_odd = 0
    index_even = index_split
    odd_block = [None] * block_number
    even_block = [None] * block_number
    print(even_block)
    async for data in inputstream_generator():
        # Add data to blocks
        data = data.flatten().tolist()
        odd_block[index_odd] = data
        even_block[index_even] = data

        # Yield blocks if full
        if index_odd == index_max:
            if None not in odd_block:
                odd_block = np.array([item for items in odd_block for item in items])
                yield odd_block
                odd_block = [None] * block_number
            index_odd = 0
        else:
            index_odd += 1
        if index_even == index_max:
            if None not in even_block:
                even_block = np.array([item for items in even_block for item in items])
                yield even_block
                even_block = [None] * block_number
            index_even = 0
        else:
            index_even += 1

async def main():
    new_block = np.array([0] * 1024)
    async for block in block_stream_generator():
        print(block)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')
