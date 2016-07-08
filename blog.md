

To design a neural network that learns from music and begins generating its own, we must first take the step from music as an audio file to music as a "tensor file", which is fancy jargon for saying "a file that a neural network can work with". It seems like a herculean task, but if done right, it could be the defining factor in the working of the neural network.

Lets start converting this raw WAV file into something that the neural network can understand.

Let us first consider a file in the monoaural WAV format. Let's break this down. We chose the WAV format because it is the most easily available decompressed audio aside from FLAC, well integrated into python (scipy.io.wavfile functions) and also allows us to capture multiple instruments and maybe even vocals (We still have to try this one out, though). We choose a monoaural audio file (Audio file with a single )

Now that we have an audio format which is decompressed AND sampled in the time domain, we can proceed.

To have the network understand the different frequencies in the time signal better, we have converted the signal from the time domain into its corresponding frequency domain using a "discrete fourier transform". "Discrete" because we have a periodically sampled time signal; and a "transform" indicating that we are converting the signal into its constitutional sine waves with their amplitudes and phases. This is probably the most important part of pre-processing the signal before feeding it into the neural network for training. We did this using the fast fourier transform (FFT) algorithm.

The output of the FFT is an array of complex numbers, which do need to be divided into real and imaginary parts before being fed into the neural network.

In order to capture sounds properly, we have to fourier transform equally spaced "buckets" of the signal so that the temporal nature of the signal is not lost. The size of these "buckets" is crucial in determining the network's ability to learn. We've set the bucket size to 11025 samples.

As for the last bucket that we have which doesn't have the necessary full number of blocks, we zero pad it at the end. This just means adding zeroes to make it divisible by our bucket size. Note that this does not change the FFT result, but makes it better descriptive because of a larger number of samples and no change in the sampling frequency.

Each of these buckets represents one pass through the network. Since we're using keras to build our model, we need to batch up the training data into blocks. So we decided to take a block to be arbitrarily equal to 40 training examples (or 40 buckets if you're counting it that way). Nothing to worry about here, just keras formalities.

Brief recap: we've taken the sample audio, divided it into buckets, disctretely fourier transformed each bucket after zero padding to ensure fitting and then just divided them further into blocks or batches.

Now, this input is fit to be fed into the neural network to train it. We'ce structured it as a 3-dimensional array. THe first dimension denotes batch size and the second, the block/bucket size.
