## Introduction ##

The songs in the training dataset are now processed and converted to a format which any neural network would understand, namely the np-tensor format. Now comes the question, "What type of neural networks should be used?". In general, there are two major variants of neural networks - the Convolutional Neural Networks and the Recurrent Neural Networks. Let's weigh up the properties of both variants and see why recurrent networks are more suitable. One observation is that the np-tensors are basically sequential information of the music.

Convolutional networks accept an input vector of fixed size and produce an output vector of fixed size. They also have limited amount of processing steps(limited by the number of hidden layers). Also, there exists no dependency between the input and output vectors. Traditionally, such networks are used for classification purposes wherein the input is converted to a np-tensor format and the output vector contains the probabilities of it being in each class. In other words, it would be a  bad idea to use convolutional networks for generating music as the output (Eg: The next note) will heavily depend on the previous sequences of notes generated. Since the music requires plausibility, we need to include the history of notes to generate the next note which is clearly not supported by convolutional networks.

## Can recurrent networks do the job for us? ##

The idea behind recurrent networks is to make use of sequential information. Recurrent neural networks are called recurrent because they repeatedly perform a same set of pre-defined operations on every element of the sequence(np-tensor in our case). The important part is that the next set of operations also takes into the account the results of previous computations. From another point of view, we can see that RNN's have a memory that can persist the information. Sounds more suitable right? We give a sequence of notes to the network, it goes through the entire sequence and generates the next note which is plausible to hear. Therefore, recurrent neural network is used.

## Understanding the structure of RNN ##

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

## How exactly does the model learn to generate music? ##

The np-tensor contains a large sequence of notes divided into single layers of a fixed length. The vector used for computing loss function is same as the input layers but shifted by 1 block. Say, L5 L4 L3 L2 L1 are the input vectors. The vectors used for computing loss function will be L6 L5 L4 L3 L2 respectively. The LSTM generates a sequence of notes which is compaed against the expected out and the errors are backpropagated thus adjusting the parameters learnt by the LSTM.
The important part is that the generated layer of notes is appended to the previous sequence thus improving the plausibility. This would have not been possible with CNNs but it is possible with RNN's as only the 3 matrices are used for computation repeatedly on the appended sequence as well!
The optimizer used is 'rms-prop' and the loss function used is 'mean squared error'.
The learnt paramets are stored in a file for generation later on.
