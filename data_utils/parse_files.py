import os
import scipy.io.wavfile as wav
import numpy as np
from pipes import quote
from config import nn_config


def convert_mp3_to_wav(filename, sample_frequency):
    # This statement gets the file extension for the file to make sure that it is an mp3 file before conversion begins
    ext = filename[-4:]

    if (ext != '.mp3'):
        return

    # The below statement splits the path for the file into an array of individual strings
    files = filename.split('/')

    # orig_filename now stores the name of the music file without the extension
    # orig_path stores the path of the mp3 file to be converted
    orig_filename = files[-1][0:-4]
    orig_path = filename[0:-len(files[-1])]

    # We define a variable new_path
    new_path = ''

    # The below statements define a value for new_path as the same folder in which the mp3 files lie
    if (filename[0] == '/'):
        new_path = '/'
    for i in range(len(files) - 1):
        new_path += files[i] + '/'

    # We now define two paths - one for the tmp folder for the mp3 files and one for the new_path which contains the WAV files
    # We also create directories if they don't already exist
    tmp_path = new_path + 'tmp'
    new_path += 'wave'

    if not os.path.exists(new_path):
        os.makedirs(new_path)
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    # We define the file names for the newly created WAV files and the already existing(?) mp3 files
    filename_tmp = tmp_path + '/' + orig_filename + '.mp3'
    new_name = new_path + '/' + orig_filename + '.wav'

    # These lines calls LAME to resample the audio file at the standard analog frequency of 44,100 Hz and then convert it to WAV
    sample_freq_str = "{0:.1f}".format(float(sample_frequency) / 1000.0)
    cmd = 'lame -a -m m {0} {1}'.format(quote(filename), quote(filename_tmp))
    os.system(cmd)
    cmd = 'lame --decode {0} {1} --resample {2}'.format(quote(filename_tmp), quote(new_name), sample_freq_str)
    os.system(cmd)

    # Returns the name of the directory where all the WAV files are stored
    return new_name


def convert_flac_to_wav(filename, sample_frequency):
    ext = filename[-5:]
    if (ext != '.flac'):
        return
    files = filename.split('/')
    orig_filename = files[-1][0:-5]
    orig_path = filename[0:-len(files[-1])]
    new_path = ''
    if (filename[0] == '/'):
        new_path = '/'
    for i in range(len(files) - 1):
        new_path += files[i] + '/'
    new_path += 'wave'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    new_name = new_path + '/' + orig_filename + '.wav'
    cmd = 'sox {0} {1} channels 1 rate {2}'.format(quote(filename), quote(new_name), sample_frequency)
    os.system(cmd)
    return new_name


# The below method converts the mp3 or FLAC files in the directory to WAV files

def convert_folder_to_wav(directory, sample_rate=44100):
    # The below for loop runs through all the mp3/FLAC files and converts them to WAV
    for file in os.listdir(directory):

        # fullfilename holds the name of the directory and the file one after another
        fullfilename = directory + file

        # The below if - elif statement converts the file to WAV based on its file extension
        if file.endswith('.mp3'):
            convert_mp3_to_wav(filename=fullfilename, sample_frequency=sample_rate)
        elif file.endswith('.flac'):
            convert_flac_to_wav(filename=fullfilename, sample_frequency=sample_rate)

    return directory + 'wave/'


def read_wav_as_np(filename):
    # wav.read returns the sampling rate per second  (as an int) and the data (as a numpy array)
    data = wav.read(filename)

    np_arr = data[1].astype('float32') / 32767.0  # Normalize 16-bit input to [-1, 1] range
    # np_arr = np.array(np_arr)
    return np_arr, data[0]


def write_np_as_wav(X, sample_rate, filename):
    Xnew = X * 32767.0
    Xnew = Xnew.astype('int16')
    wav.write(filename, sample_rate, Xnew)
    return


def convert_np_audio_to_sample_blocks(song_np, block_size):  # this returns song_np by padding it

    # Block lists initialised
    block_lists = []

    # total_samples holds the size of the numpy array
    total_samples = song_np.shape[0]

    #num_samples_so_far is used to loop through the numpy array
    num_samples_so_far = 0

    while (num_samples_so_far < total_samples):

        # Stores each block in the "block" variable
        block = song_np[num_samples_so_far:num_samples_so_far + block_size]

        if (block.shape[0] < block_size):
            padding = np.zeros(
                    (block_size - block.shape[0],))  # this is to add 0's in the last block if it not completely filled
            block = np.concatenate((block,
                                    padding))  # block_size is 11025 which is fixed throughout whereas block.shape[0] for the last block is <=11025
        block_lists.append(block)
        num_samples_so_far += block_size
    return block_lists



def convert_sample_blocks_to_np_audio(blocks):
    song_np = np.concatenate(blocks)
    return song_np


def time_blocks_to_fft_blocks(blocks_time_domain):
    fft_blocks = []
    for block in blocks_time_domain:
        # Computes the one-dimensional discrete Fourier Transform and returns the complex nD array
        # i.e The truncated or zero-padded input, transformed along the axis indicated by axis, or the last one if axis is not specified.
        fft_block = np.fft.fft(block)
        new_block = np.concatenate(
                (np.real(fft_block), np.imag(fft_block)))  # Joins a sequence of arrays along an existing axis.
        fft_blocks.append(new_block)
    return fft_blocks


def fft_blocks_to_time_blocks(blocks_ft_domain):
    time_blocks = []
    for block in blocks_ft_domain:
        num_elems = block.shape[0] / 2
        real_chunk = block[0:num_elems]
        imag_chunk = block[num_elems:]
        new_block = real_chunk + 1.0j * imag_chunk
        time_block = np.fft.ifft(new_block)
        time_blocks.append(time_block)
    return time_blocks


def convert_wav_files_to_nptensor(directory, block_size, max_seq_len, out_file, max_files=20, useTimeDomain=False):
    files = []

    # If the file is already a WAV file, then the code simply stores it as it is
    for file in os.listdir(directory):
        if file.endswith('.wav'):
            files.append(directory + file)

    # chunks_X and chunks_Y are initialised as lists
    chunks_X = []
    chunks_Y = []

    # The code takes in a maximum of 20 files and if greater, then the first twenty alone
    num_files = len(files)
    if (num_files > max_files):
        num_files = max_files

    # This loops through the indices (0 -> max_files) of the files list
    for file_idx in range(num_files):
        # Each file is stored in the variable "file"
        file = files[file_idx]

        # Prints some sort of processing message to the user, using file index and number of files
        print('Processing: ', (file_idx + 1), '/', num_files)
        print('Filename: ', file)


        X, Y = load_training_example(file, block_size, useTimeDomain=useTimeDomain)
        cur_seq = 0
        total_seq = len(X)
        print(total_seq)
        print(max_seq_len)
        while cur_seq + max_seq_len < total_seq:
            chunks_X.append(X[cur_seq:cur_seq + max_seq_len])
            chunks_Y.append(Y[cur_seq:cur_seq + max_seq_len])
            cur_seq += max_seq_len
    num_examples = len(chunks_X)
    num_dims_out = block_size * 2
    if (useTimeDomain):
        num_dims_out = block_size
    out_shape = (num_examples, max_seq_len, num_dims_out)
    x_data = np.zeros(out_shape)
    y_data = np.zeros(out_shape)
    for n in range(num_examples):
        for i in range(max_seq_len):
            x_data[n][i] = chunks_X[n][i]
            y_data[n][i] = chunks_Y[n][i]
        print('Saved example ', (n + 1), ' / ', num_examples)
    print('Flushing to disk...')
    mean_x = np.mean(np.mean(x_data, axis=0), axis=0)  # Mean across num examples and num timesteps
    std_x = np.sqrt(
        np.mean(np.mean(np.abs(x_data - mean_x) ** 2, axis=0), axis=0))  # STD across num examples and num timesteps
    std_x = np.maximum(1.0e-8, std_x)  # Clamp variance if too tiny
    x_data[:][:] -= mean_x  # Mean 0
    x_data[:][:] /= std_x  # Variance 1
    y_data[:][:] -= mean_x  # Mean 0
    y_data[:][:] /= std_x  # Variance 1

    np.save(out_file + '_mean', mean_x)
    np.save(out_file + '_var', std_x)
    np.save(out_file + '_x', x_data)
    np.save(out_file + '_y', y_data)
    print('Done!')


def convert_nptensor_to_wav_files(tensor, indices, filename, useTimeDomain=False):
    num_seqs = tensor.shape[1]
    for i in indices:
        chunks = []
        for x in range(num_seqs):
            chunks.append(tensor[i][x])
        save_generated_example(filename + str(i) + '.wav', chunks, useTimeDomain=useTimeDomain)


def load_training_example(filename, block_size=2048, useTimeDomain=False):

    #read_wav_as_np returns data as a numpy array and the sampling rate stored in data and bitrate respectively
    data, bitrate = read_wav_as_np(filename)

    # x_t has the padded data i.e with 0's in the empty space of the last block
    x_t = convert_np_audio_to_sample_blocks(data, block_size)


    y_t = x_t[1:]
    y_t.append(np.zeros(block_size))  # Add special end block composed of all zeros
    if useTimeDomain:
        return x_t, y_t
    X = time_blocks_to_fft_blocks(x_t)
    Y = time_blocks_to_fft_blocks(y_t)
    return X, Y


def save_generated_example(filename, generated_sequence, useTimeDomain=False, sample_frequency=44100):
    if useTimeDomain:
        time_blocks = generated_sequence
    else:
        time_blocks = fft_blocks_to_time_blocks(generated_sequence)
    song = convert_sample_blocks_to_np_audio(time_blocks)
    write_np_as_wav(song, sample_frequency, filename)
    return


def audio_unit_test(filename, filename2):
    data, bitrate = read_wav_as_np(filename)
    time_blocks = convert_np_audio_to_sample_blocks(data, 1024)
    ft_blocks = time_blocks_to_fft_blocks(time_blocks)
    time_blocks = fft_blocks_to_time_blocks(ft_blocks)
    song = convert_sample_blocks_to_np_audio(time_blocks)
    write_np_as_wav(song, bitrate, filename2)
    return
