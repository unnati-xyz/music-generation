from data_utils.parse_files import *
import config.nn_config as nn_config

config = nn_config.get_neural_net_configuration()

# Here, the freq, input_directory and output_filename are initialized as per the configured neural network

input_directory = config['dataset_directory']
output_filename = config['model_file'] 

freq = config['sampling_frequency'] #sample frequency in Hz

# The author defines "clip_len", "block_size" and "max_seq_len"

clip_len = 10 		#length of clips for training. Defined in seconds
block_size = freq / 4 #block sizes used for training - this defines the size of our input state
max_seq_len = int(round((freq * clip_len) / block_size)) #Used later for zero-padding song sequences
#Step 1 - convert MP3s to WAVs
new_directory = convert_folder_to_wav(input_directory, freq)
#Step 2 - convert WAVs to frequency domain with mean 0 and standard deviation of 1
convert_wav_files_to_nptensor(new_directory, block_size, max_seq_len, output_filename)
