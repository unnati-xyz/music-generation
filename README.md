# music-generation

Algorithmic music generation using RNN(Recurrent Neural Networks)

Language used : Python 3.5

The following packages has to be installed before running this code:

1. *Keras* version 0.1.0 with Theano as the backend.
2. *NumPy* and *SciPy* for various mathematical computation on tensors.
3. *Matplotlib* for visualizing the input.
4. *LAME* and *SoX* to convert mp3 files into other formats such as wav.

## Step 1: Converting the given mp3 files into np tensors

Type the following command into the terminal:

``python convert_directory.py``

This converts mp3 into mono files and then into WAV file, which is stored in the form of np-tensors. These np-tensors are given as input to our LSTM  model.
By the end of this one can find these 2 files generated "YourMusicLibraryNP_x.npy", "YourMusicLibraryNP_y.npy". 
"YourMusicLibraryNP_x.npy" contains the input sequence for training and "YourMusicLibraryNP_y.npy" contains the same sequence as that of input sequence but shifted by one block.

##Step 2: Training the model

Type the following command into the terminal:

``python train.py``

This builds a LSTM model that generates a sequence of notes which is compared against the expected output and the errors are back-propagated, thus adjusting the parameters learnt by the LSTM. 
You can change the number of Iterations, number of epochs per iteration and batch size by adjusting the following parameters "num_iters", "epochs_per_iter", "batch_size" respectively in train.py.

##Step 3: Generating the music

Now that you've finished training the model, its time to generate some music:)
Type the following command in your terminal':

``python generate.py``

The generated WAV file is stored in a file named generated_song.wav. 

You can further read about the entire procedure involved in generating the music, in the following post that is there on this repository, **Blog_Post_Music_Gen_1.md**.

