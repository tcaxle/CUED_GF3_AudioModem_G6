from . import imports,exports
from . import OFDM_block_creation as ofdm

def transmit(input_file="input.txt", input_type="txt", save_to_file=False, suppress_audio=False):
    """
    Parameters
    ----------
    input_file : STRING
        name of the input file
    input_type : STRING
        "txt" for text input
        "wav" for wav input
    save_to_file : BOOL
        if set then outputs data "output.txt"
    suppress_audio : BOOL
        if set then does not output sound
    """
    if input_type == 'txt':
        
        data = imports.text_to_binary()
        data = ofdm.fill_binary(data)
        data = ofdm.xor_binary_and_key(data)
        data = ofdm.binary_to_words(data)
        data = ofdm.words_to_constellation_values(data)
        data = ofdm.constellation_values_to_data_blocks(data)
        data = ofdm.assemble_block(data)
        data = ofdm.block_ifft(data)
        data = ofdm.cyclic_prefix(data)
        preamble = ofdm.create_preamble()
        data = [preamble] + data
        data = exports.play_output(data)
    
    if input_type == 'wav':
        
        data = imports.wav_to_binary()
        data = ofdm.binary_to_words(data)
        data = ofdm.words_to_constellation_values(data)
        data = ofdm.constellation_values_to_data_blocks(data)
        data = ofdm.assemble_block(data)
        data = ofdm.block_ifft(data)
        data = ofdm.cyclic_prefix(data)
        preamble = ofdm.create_preamble()
        data = [preamble] + data
        data = exports.play_output(data)
    
    pass