<h3>Can computers be creative enough to make music ?</h3>
In the recent times, engineers & researchers have been extremely succesful in using Machine Learning for speech recognition, image classification,predictive analysis etc and have developed models which can achieve human level accuracy(or more) in the former tasks, but can we use Machine Learning to make generative models which can generate music,text,art etc.
Turns out, computers can actually churn out meaningful <a href="http://karpathy.github.io/2015/05/21/rnn-effectiveness/">text</a> and <a href="https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html">sensible images</a>.

Over the past 2 months we attempted to generate music using deep learning techniques.We used the LSTM(Long Short Term Memory) flavour of RNNs.

<p>
 Before we go on to construct a neural network, we need to a have a dataset on which we can train the network.So, the first step was to convert music files(which are usually in mp3 format) into a format which the neural network can understand.This innvolved some digital signal processing stuff.
</p>

<p>
We converted raw mp3 files to monaural wav format.
A monaural sound uses just one channel and the same channel is sent to all the speakers. So, the sound we hear in both our ears is exactly identical. As a result of this monaural sounds do not have directionality/spaciality which is not much of a requirement to generate music(it's more relevant in live perfomances etc).
Also, the numpy arrays produced from monoaural songs is half the size of that produced by stero sounds. This reduces memory requirements as well as the time taken to train our neural network.
</p>

<p>
We used <a href="http://lame.sourceforge.net/">lame</a> which is an open source mp3 encoder to convert the monaural files to wav format.
We preferred monoaural wav files over monoaural mp3 files, even though wav files being uncompressed occupy more memory because there are functions in <a href="http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.io.wavfile.read.html">scipy</a> which can read and write wav files directly into numpy arrays.There aren't any reliable open source packages which can read/write mp3 files to numpy arrays because of various <a href="https://github.com/scipy/scipy/issues/3536">copyright & patent issues</a> assosciated with the mp3 format.
The values in the numpy arrays are normalized to [-1,1] range in order to increase the speed of gradient descent.
</p>

<p>
We know that sound waves are continous signals(infinite datapoints).However, our computers only operate on discrete values.Yet, computers can store and play music due to sampling.In sampling, we store the value of the signal at regular intervals of time determined by the sampling frequency(finite datapoints).
We have set the sampling frequency to 44100 Hz.Human ear is senstitive only to frequencies upto 20,000 Hz. So, even if frequencies above 20,000 Hz are present in the song, they do not make a difference as they are inaudible. So, according to Nyquist's Sampling Theorem the sampling frequency must be nearly double that of 20,000 Hz. So, 44100 Hz is considered the standard sampling frequency.48000 Hz is another standard sampling frequency.


We then divided the numpy arrays of the wav files to smaller blocks each of size 44100 and zero padded the last block,so that it also has a size of 44100.Zero padding does not alter the frequency content of the signal and does not increase the resolution of the discrete signal given as output by the Fourier Transform. It just increases the number of phasors outputted by the fourier transform.FFT(Fast Fourier Transform)as the name suggests is a fast version of fourier transform and is most efficient when the input size is a power of of 2. So, the default block size is fixed to 2048.

 </p>

<h5>How does algorithmic music generation help?</h5>
<ol>
<li> Can assist music composers </li>

<li> Used in Youtube videos,games etc </li>

<li> Helps us understand the power & limitations of Machine Learning techniques</li>

</ol>

<h5>Problems we faced</h5>
<ol>
<li> Takes alot of time to train to the network making it difficult to experiment with different configurations</li>

<li> We didn't have a suitable mechanism to validate the  generated music, we just went by intution</li>

</ol>

<h5>Future Plans</h5>
<ol>

<li>Try to generate music with lyrics :p </li>

<li>Mix different instruments & genres of music</li>

</ol>

The entire code for our project is open sourced and is available on <a href="https://github.com/unnati-xyz/music-generation">GitHub</a>.Our program learns to generate music from raw mp3 files. So, you can use sufficient **mp3 files of your choice as training data** to make the kind of music you like!

<h5>References</h5>
<ol>
<li><a href="http://jackschaedler.github.io/circles-sines-signals/index.html">Digital Signal Processing</a>
<li><a href="http://colah.github.io/posts/2015-08-Understanding-LSTMs/">LSTMs</a>
<li><a href="https://cs224d.stanford.edu/reports/NayebiAran.pdf">GRUV:Algorithmic Music Generation using RNN</a>
</ol>
