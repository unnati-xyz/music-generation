# Time to generate some Music!

So now that we've already seen how the sampling of waves and the training of model using LSTM networks is done in the earlier sections, let's see how exactly the music is generated using this trained model.

Once the model is trained, we can generate music with this model by presenting it with the first few notes of a song that it has never seen in training, which is known as seed sequence.

Here's the interesting part, we need to create some seed sequence for the algorithm to start with. Currently, we just grab an existing seed sequence from our training data and use that. However, this will generally produce verbatum copies of the original songs. In a sense, choosing good seed sequences reflects how you get interesting compositions. There are many, many ways we can pick these seed sequences such as taking linear combinations of certain songs. We could even provide a uniformly random sequence, but that is highly unlikely to produce good results.

Now coming back to our earlier point of presenting the trained model with the seed sequence, once it is done, we can use the network predictions to generate network inputs. Network predictions are conditioned using a softmax function, ensuring that the sum of the output vector is 1.0. This allows us to interpret the output vector as a probability estimation from which we can select the next note. The selected note is then presented to the network at the next timestep as an input. 

In nutshell, here is the generation algorithm:
**Step 1** - Given A = [X\_0, X\_1, ... X\_n], generate X\_n + 1.
**Step 2** - Concatenate X\_n + 1 onto A.
**step 3** - Repeat the entire procedure MAX\_SEQ\_LEN times.
In this project **MAX\_SEQ\_LEN** is nothing but **freq \* clip\_len) / block\_size**.
Where **freq**=44100 Hz, **clip\_len**=10 seconds and **block\_size=freq/4**=11025.

Voila! After these 3 simple steps, you've your algorithmically generated music ready! 

## Conclusion

We've seen that algorithmic music generation with waveforms as input is possible with the use of recurrent neural networks, particularly the LSTM network. Further we can investigate the effect of adding layers of recurrent units and discovering the impact that additional layers have on performance, performing network training across genres with a substantially larger corpus.  

https://soundcloud.com/padmaja-bhagwat/generated-music

[Here](https://soundcloud.com/padmaja-bhagwat/generated-music) is the link to the music that we generated for the first time after training the model using 20 different piano songs for about 2000 iterations. **Enjoy:)**