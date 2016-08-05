<h2>Can computers be creative enough to make compelling music ?</h2>
![Image of Robot playing music](http://www.i-programmer.info/images/stories/News/2014/Nov/A/Naomusicicon.jpg)
<p>
Most of you might have heard about how Deep Mind's computer program *AlphaGo* defeated the world champion Lee Sedol three times in a row.A typical game of Go has about 10<sup>360</sup> moves,a humongous number.So, the program did not win the game by brute force , but it actually thought of a stratergy before making a move, just like we do.*AlphaGo* managed to achieve this feat just by using Deep Learning combined with some spectacular software engineering.
Recently, Deep Learning has also been extremely succesful  in classifying images,recognising human speech and making predictions at human level accuracy(or more).Clearly, Deep Learning is really good at doing things like us humans! An interesting problem is can we extend Deep Learning to make generative models which can generate pieces of art and music like artists ?
Turns out, computers can actually churn out meaningful <a href="http://karpathy.github.io/2015/05/21/rnn-effectiveness/">text</a> and <a href="https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html">sensible images</a> with a little bit of training.

Over the past 2 months we at Unnati Data Labs,attempted to generate music using algorithms based on Deep Learning techniques.We used the LSTM(Long Short Term Memory) flavour of Recurrent Neural Networks(don't get bogged down by the fancy name) to accomplish this task.In this blog posts we'll explain how we tackled the problem with minimal technical jargon :p

You might wonder why generating music using computer programs is a good idea.Well, because
<ol>
<li> It can assist music composers.Musiscians might come up with unique ways to use music generating tools. </li>

<li> Algorithmically generated music can be used in youtube videos, games, movies etc </li>

<li> It helps us understand the power & limitations of Machine Learnin & Deep Learning techniques.It helps to see wether computer programs can be creative and innovative.</li>

</ol>

Before, we start off you can listen to the music we generated at the end of our experiment [here](https://soundcloud.com/padmaja-bhagwat/generated-music).

<p>
Firstly, we need to a have a dataset on which we can train our neural network.So, the first step was to convert music files(which are usually in mp3 format) into a format which the neural network can understand.The input to neural networks are tensors which are just multi dimensional arrays.Hence, we had to convert the audio files to tensors.This innvolved some<a href="http://jackschaedler.github.io/circles-sines-signals/index.html"> digital signal processing</a> stuff.
This might seem like a herculean task,but if done right, it could be the defining factor in the working of the neural network.
</p>

<p>
Let's spend some time understanding the type of data we are dealing with.We know that sound waves are continous signals(infinite datapoints).However, our computers only operate on discrete values.Yet, computers can store and play music due to sampling.In sampling, we store the value of the signal at regular intervals of time determined by the sampling frequency(finite datapoints).
Sampling does lead to losing some data, but we cannot percieve the loss.
We have set the sampling frequency to 44100 Hz.Human ear is senstitive only to frequencies upto 20,000 Hz. So, even if frequencies above 20,000 Hz are present in the song, they do not make a difference as they are inaudible. So, according to Nyquist's Sampling Theorem the sampling frequency must be nearly double that of 20,000 Hz. So, 44100 Hz is considered the standard sampling frequency.48000 Hz is another standard sampling frequency.
</p>

<h4>How to feed songs to the Neural Network ?</h4>
<p>
Initially we converted raw mp3 files to monaural wav format.
A monaural sound uses just one channel and the same channel is sent to all the speakers. So, the sound we hear in both our ears is exactly identical. As a result of this monaural sounds do not have directionality/spaciality which is not much of a requirement to generate music(it's more relevant in live perfomances etc).
Also, the numpy arrays produced from monoaural songs is half the size of that produced by stero sounds. This reduces memory requirements as well as the time taken to train our neural network.
</p>

<p>
We used <a href="http://lame.sourceforge.net/">lame</a> which is an open source mp3 encoder to convert the monaural files to wav format.
We preferred monoaural wav files over monoaural mp3 files, even though wav files being uncompressed occupy more memory because the WAV format is well integrated with python( <a href="http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.io.wavfile.read.html">scipy functions</a>) .There aren't any reliable open source packages which can process mp3 files because of various <a href="https://github.com/scipy/scipy/issues/3536">copyright & patent issues</a> assosciated with the mp3 format.

</p>

<p>
To have the network understand the different frequencies in the time signal better, we have converted the signal from the time domain into its corresponding frequency domain using a "Discrete Fourier Ttransform".Let's try to break down this long term. "Discrete" because we have a periodically sampled time signal; and a "transform" indicating that we are converting the signal into its constitutional sine waves with their amplitudes and phases. This is probably the most important part of pre-processing the signal before feeding it into the neural network for training. We did this using the Fast Fourier Transform (FFT) algorithm.

The output of the FFT is an array of complex numbers, which do need to be divided into real and imaginary parts before being fed into the neural network.

In order to capture sounds properly, we have to fourier transform equally spaced "buckets" of the signal so that the temporal nature of the signal is not lost. The size of these "buckets" is crucial in determining the network's ability to learn. We've set the bucket size to 11025 samples.


Each of these buckets represents one pass through the network. Since we're using keras to build our model, we need to batch up the training data into blocks. So we decided to take a block to be arbitrarily equal to 40 training examples (or 40 buckets if you're counting it that way). Nothing to worry about here, just keras formalities.

Brief recap: We've taken the sample audio in WAV format, divided it into buckets, disctretely fourier transformed each bucket after zero padding to ensure fitting and then just divided them further into blocks or batches.

Now, this input is fit to be fed into the neural network to train it. We've structured it as a 3-dimensional array. The first dimension denotes batch size and the second, the block/bucket size.
</p>

<h4>Which Neural Network Architecture to use ?</h4>
<p>
Now, we are done with all the pre-processing tasks.The question which arises now is , "What type of neural networks should be used?". In general, there are two major variants of neural networks - the Convolutional Neural Networks and the Recurrent Neural Networks. Let's weigh up the properties of both variants and see why recurrent networks are more suitable for the task at hand. One observation is that the np-tensors we have are basically sequential information of the music.

Convolutional networks accept an input vector of fixed size and produce an output vector of fixed size. They also have limited amount of processing steps(limited by the number of hidden layers). Also, there exists no dependency between the input and output vectors. Traditionally, such networks are used for classification purposes wherein the input is converted to a np-tensor format and the output vector contains the probabilities of it being in each class. In other words, it would be a  bad idea to use convolutional networks for generating music as the output (Eg: The next note) will heavily depend on the previous sequences of notes generated. Since the music requires plausibility, we need to include the history of notes to generate the next note which is clearly not supported by convolutional networks.
</p>

<p>
Let's try to see if recurrent neural networks can do the job for us? 
The idea behind recurrent networks is to make use of sequential information. Recurrent neural networks are called recurrent because they repeatedly perform a same set of pre-defined operations on every element of the sequence(np-tensor in our case). The important part is that the next set of operations also takes into the account the results of previous computations. From another point of view, we can see that RNN's have a memory that can persist the information. Sounds more suitable right? We give a sequence of notes to the network, it goes through the entire sequence and generates the next note which is plausible to hear. Therefore, recurrent neural network is used.
</p>

<h4>Understanding Recurrent Neural Networks</h4>
<p>
Recurrent neural networks have loops in them thus allowing persistence of information. Loops can be visuzalized as a layer having sequential neurons wherein each neuron accepts the input from previous layer as well as from previous neuron in the same layer.
![Visualizing RNN as an unfolded layer of neurons](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)

This way of visualization shows the degree of aptness between sequences and recurrent neural networks.
But there is a drawback of vanilla recurrent neural networks, they cannot persist the information for long periods of time. A slightly complex model of vanilla recurrent neural networks is known as LSTM(Long Short Term Memory). A separate vector is dedicated for persisting the information known as the cell state.

![Structure of LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

One huge advantage of LSTM's is that the number of parameters that it needs to learn is less compared to traditional networks. There are basically 3 matrices acting as weights for carrying forward information, updating information and producing output. The same 3 matrices are repeatedly used to perform operations on each element of the sequence. Let us look at each one of them.

![Cell state](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-C-line.png)

![Step1](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png)

Assume we are at step t, C denotes cell state, h denotes the output and x denotes input. We need to first decide how much of the previous information to persist based on the current input and previous output. This decision is made by a sigmoid layer that outputs a number between 0 and 1. A number closer to 1 is an indication of persistence.

![Step2A](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png)
![Step2B](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-C.png)

The next step is to update the cell state. To do this, we need to include current input and previous output in the computations. Both vectors are passed through a tanh layer of neurons and sigmoid layer of neurons to extract the new information and scale the amount of new information to be updated respectively.

![Step3](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png)

The last step is to produce the output. Output depends on current cell state (updated version). The updated cell state vector is passed through a tanh layer and is scaled by current input and previous output passed through a sigmoid layer of neurons.

For more info:http://colah.github.io/posts/2015-08-Understanding-LSTMs

The architecture of the LSTM used for music generation is a shallow network consisting of just 1 recurrent unit. The input and output neuron layers have the same size as the size of the np-tensor. The single hidden layer consists of 1024 neurons. We are still experimenting to find a better architecture by making the network denser. The shallow network requires around 2000 iterations for generating plausible music. Hopefully we will require lesser number of iterations on making the network denser while maintaining plausibility.
</p>

<h4>How exactly does the model learn to generate music? </h4>

<p>
The np-tensor contains a large sequence of notes divided into single layers of a fixed length. The vector used for computing loss function is same as the input layers but shifted by 1 block. Say, L5 L4 L3 L2 L1 are the input vectors. The vectors used for computing loss function will be L6 L5 L4 L3 L2 respectively. The LSTM generates a sequence of notes which is compaed against the expected out and the errors are backpropagated thus adjusting the parameters learnt by the LSTM.
The important part is that the generated layer of notes is appended to the previous sequence thus improving the plausibility. This would have not been possible with CNNs but it is possible with RNN's as only the 3 matrices are used for computation repeatedly on the appended sequence as well!


In nutshell, here is the generation algorithm:

**Step 1** - Given A = [X<sub>0</sub>, X<sub>1</sub>, ... X<sub>n</sub>], generate X<sub>n + 1</sub> ( Regresssion Problem )

**Step 2** - Concatenate X<sub>n + 1</sub> onto A.

**Step 3** - Repeat the entire procedure MAX\_SEQ\_LEN times.

In our project **MAX\_SEQ\_LEN** is nothing but **(freq \* clip\_len) / block\_size**.
Where **freq**=44100 Hz, **clip\_len**=10 seconds and **block\_size=freq/4**=11025.


<ol>
We did face some problems during the course of our project.
<li>We didn't have a suitable mechanism to validate the generated music, we just went by intution.It is quite difficult to define "good music".</li>
<li>It takes quite alot of time to train to the network making it difficult to experiment with different configurations.We used an AWS instance with 16GB RAM.It took 3 days to complete 2000 iterations on just 10 songs each about a minute long.</li>
<ol>

</p>


<p>
In case you are interested, the entire code for our project is open sourced and is available on <a href="https://github.com/unnati-xyz/music-generation">GitHub</a>.Our program learns to generate music from raw mp3 files. So, you can use sufficient mp3 files of your choice as training data to make the kind of music you like!
</p>


