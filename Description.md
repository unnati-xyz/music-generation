# Music Generation

This project aims to create music using Neural Networks and Deep Learning.

It learns to generate music from raw audio files. So, you can use sufficient **mp3 files of your choice as training data**

Language Used : Python 3.5

Dependancies  : Numpy, Scipy, Theano, Keras 0.1.0, lame mp3 encoder


### A) **Converting mp3 into wav**

 **A.1) Converting the given mp3 into a monaural sound**
 
A monaural sound uses just one channel and the same channel is sent to all the speakers. So, the sound we hear in both our ears is exactly identical. As a result of this monaural sounds do not have directionality/spaciality which is not much of a requirement to generate music.
Also, the numpy arrays produced from monoaural songs is half the size of that produced by stero sounds. This reduces memory requirements as well as the time taken to train our neural network.

**A.2) Converting monaural mp3 into wav format**

The mono files obtained in A.1 are then converted to wav format using lame which is an open source mp3 encoder.The wav files obtained are stored in the directory YourMusicLibrary.

### B) **Converting wav file into nptensor**

**B.1) Wav file to numpy array**

The program takes in each of the wav files in the directory YourMusicLibrary and reads them into numpy arrays using the function wav.read() in the scipy package.
The numpy array (16 bit integers) is then normalized to [-1, 1] range in order to increase the speed of gradient descent(feature scaling).

**B.2) Sampling Frequency**

For further analysis, we have to convert the continous audio signal with infinite data points to a discrete signal with finite number of data points. So, we sample the audio signal at regular intervals of time depending on the sampling frequency.The sampling frequency is set to 44100 Hz.Human ear is senstitive only to frequencies upto 20,000 Hz. So, even if frequencies above 20,000 Hz are present in the song, they do not make a difference as they are inaudible. So, according to Nyquist's Sampling Theorem the sampling frequency must be nearly double that of 20,000 Hz. So, **44100 Hz** is considered the standard sampling frequency.

**B.3) Breaking down the array to smaller blocks**

The numpy array of each wav file is then divided to smaller blocks each of size 44100 and the last block is zero padded,so that it also has a size of 44100.**Zero padding does not alter the frequency content of the signal** and does not increase the resolution of the discrete signal given as output by the Fourier Transform. It just increases the number of phasors outputted by the fourier transform.FFT(Fast Fourier Transform)as the name suggests is a fast version of fourier transform and is most efficient when the input size is a power of of 2. So, the default block size is fixed to 2048.

**B.4) Time Domain to Frequency Domain using Fourier Transform**

The current time domain wave is converted to freqency domain using Fast Fourier Transform.The blocks obtained in step B.2 are given as input to the function performing FFT.
The function np.fft.fft() which is a part of the numpy package returns  a nDarray of complex numbers.The complex part of the returned complex numbers is seperated from the real part and is appended further, which results in doubling of the total block size from 44100 to 88200.

**B.5) Convert the input matrix to 3D**

X in the program is a 2D matrix, it is then converted into a 3D matrix namely chunks_x by further dividing the x-axis into the blocks of max_seq_len(10) each.This gives us a sequence of inputs to feed to the RNN.chunks_x is the nptensor which is given as input to our RNN.

**B.6) Save input & target tensors**

The matrices x_data and y_data are just another copy of chunks_x and chunks_y respectively.The 	matrices x_data and y_data are stored YourMusicLibraryNP_x.npy and YourMusicLibraryNP_y.npy respectively. So, we have to run convert_directory.py only once for each dataset. 

**B.7) Normalization(Data-centering process)**

Mean and variance of the elements in the matrix X and Y is calculated and are used for normalization.The Mean & Variance are saved as       				YourMusicLibraryNP_mean.npy and YourMusicLibraryNP_var.npy respectively.These values are again used 	in the end so we save them instead of re-computing them.The tensor outputted by the neural network will contain values in the range of [-1,1] due to the feature scaling we did initially. So, we use 		the mean & variance to undo the feature scaling and find valid frequencies.

### **C) The Neural Network**

**C.1) Type of Neural Network**

We want the neural network to generate songs given some songs as training data.
    	The above problem can also be stated as follows : Given a set of vectors X0, X1, X2.....Xt representing the audio waveforms at time intervals 0,1,2....t respectively, generate a vector Xt+1 for the next time interval t+1. So, given a sequence of inputs, we want to the neural network to predict the next element in the sequence.RNNs(Recurrent Neural Networks) work best for sequence modelling tasks with variable length input and output.
    	The initial sequence which is given to the network for generating a song, is a small portion(seed_len)of a randomly chosen training example.This is done in seed_generator.py.
    	The program uses the **LSTM(Long Short Term Memory)** flavour of RNN which works best for modelling long term dependencies.The network may have to remember the first tone in the sequence to generate good music.However,if we use plain RNN then, the value which is outputted by a node at one time stamp becomes the input to the same node at the next time stamp but is then lost unless the same value is outputed again.Hence, we use LSTMs.


**C.2) Architecture of the Neural Network**

As of now,the neural net is a **linear shallow** one(just one hidden layer).
    	The input layer takes in 44100 sequences and outputs 1024 sequences which are fed into the single LSTM neuron in the hidden layer.The output layer outputs 44100 sequences.
    	We want to construct a LSTM RNN of **many to many architecture**.Suppose our input sequence is ( a1, a2, a3, a4,...an) then we want an output sequence say ( b1, b2, b3, b4,... bn) such that if we push a1 to the network, we should get b1,then on pushing a2 and b1 to the network, we must get (b1, b2) and so on. To accomplish this,we use Time Distributed Dense Layers of keras. In Time Distributed Dense the same output activation function is applied to every timestep of a 3D tensor( X_train is a 3D tensor with the 3rd dimension sorresponding to time).So, the output function is computed when we give a1 as input, when we give a1 & b1 as input and so on i.e, at each unfolding of the RNN. In the LSTM Layer we set the return_sequences parameter to True.This ensures that the network outputs the entire sequence, else it will output only the last predicted term of the sequence( bn in our example) and the loss function will be computed only for the last term in the sequence & not all the terms of the output sequence.

**C.3) Training the Neural Network**

The expected output sequence for the training data is stored in the matrix Y.Y is same as that of X, translated by one block and has an extra zero padded block at the end.Y is further converted into 3D matrix called chunks_y.
The mean squared error(L2 Distance)with Y as the target output is used as the cost function for training.
 The optimizer RMSprop(Root Mean Square) is used with all the default values(including the learning rate) itself as it is the recommended optimizer for RNNs.
        #Write about the box and it's configuration etc
        #Write number of iterations,epochs,batch size,hours of training etc

The program does not perform any validation as the main intention was to generate good music alone.


