#!/usr/bin/env python3
"""
Module for listening to microphone and taking inputs. Generally uses
asynio to process and listen concurrently.

Requires: python>=3.7
"""
# Imports
import asyncio
import queue
import sys
import numpy as np
import sounddevice as sd

# Globals
SAMPLE_BLOCK_LENGTH = 265
SAMPLE_RATE = 44100

async def inputstream_generator(channels=1):
    """
    Generator that yields blocks of input data as NumPy arrays.
    Taken from examples in the sounddevice documentation.
    """
    q_in = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy()))

    stream = sd.InputStream(callback=callback, channels=channels, blocksize=SAMPLE_BLOCK_LENGTH, samplerate=44100)
    with stream:
        while True:
            data = await q_in.get()
            yield data

async def block_stream_generator(block_number = 2):
    """
    Takes blocks from inputstream_generator()
    and a block_number, which is the number of 2*SAMPLE_BLOCK_LENGTH
    sample_blocks you want in an output block.

    Returns blocks where the first half of any block is the last half
    of the previous one. The idea being that anything you are looking for
    will be guaranteed to be wholy contained within a block.
    """
    # Double the block_number to get number of sample blocks per output block
    block_number = block_number * 2
    # Maximum loop index
    index_max = int(block_number - 1)
    # Loop index for the even blocks to start from
    index_split = int(block_number / 2)
    # Initialise loop indices
    index_odd = 0
    index_even = index_split
    # Initalise output blocks
    odd_block = [None] * block_number
    even_block = [None] * block_number
    async for data in inputstream_generator():
        # Flatten sample blocks and convert them to regular lists
        data = data.flatten().tolist()
        # Add data to blocks
        odd_block[index_odd] = data
        even_block[index_even] = data

        if index_odd == index_max:
            # Yield blocks if full
            if None not in odd_block:
                odd_block = np.array([item for items in odd_block for item in items])
                yield odd_block
                odd_block = [None] * block_number
            # Reset the index
            index_odd = 0
        else:
            # Increment the index
            index_odd += 1
        if index_even == index_max:
            # Yield blocks if full
            if None not in even_block:
                even_block = np.array([item for items in even_block for item in items])
                yield even_block
                even_block = [None] * block_number
            # Reset the index
            index_even = 0
        else:
            # Increment the index
            index_even += 1

async def main():
    async for block in block_stream_generator():
        print(block)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')
