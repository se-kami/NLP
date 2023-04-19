# NLP
Simple NLP problems solved using pytorch and torchtext.

## RNN
### Counting
RNN models keep memory so they can be used for counting, for example keep track of number of consonants in a sentence.
Memory increases by 1 each time a consonant appears.

### Increase all digits by 1
Given a sequence of digits, increase all digits by 1 (9 maps to 0).
Because this problem requires no memory of previous state, this can be done in a simpler way by using a simple FC network,
but it's nice to see that RNNs can solve even these kinds of problems.

## Character level model
Each letter is a separate token.

### Name classification
Find the nationality from a person's name, using either RNN or Transformer.

1. Split name into characters
2. Add a special <eos> token to the end
3. Feed the next character into the model until <eos> is reached.
4. Pass the output through a FC network to get probability distribution over countries

To speed up the computation dynamically pad examples in a batch and feed them into the network all at once.
Then extract the output of the <eos> token.

### Name generation
Given a language and a starting string, generate a name, using either LSTM or Transformer.

To train the decoder model:

1. Split name into characters
2. Add a special <start_country> token to the beggining (different for each coutry)
3. Add a special <eos> token to the end
4. Feed the character into the model until and take the output
5. Pass the output through a FC network to get a probability distribution over possible next characters
6. Use the distribution and the true next character to calculate the cross entropy loss.
7. Backpropagate to update the network


## Translation

### Data
Raw data is in form of pair of sentences in source and target language.
To convert it into something that the transformer can read perform:
1. Tokenize - split text into units and normalize the units
2. Add special tokens - add SOS (start of sequence) and EOS (end of sequence) tokens to text
3. Build a vocabulary - go through text and count number of tokens, then replace top N (10000) tokens with an integer index and replace other tokens with a default index (UNK token)
4. Make batches - to efficiently pass the input through the transformer all examples should have the same length, add padding token to all examples in the batch

#### Example
Raw strings:
```
I am tall
I was hungry yesterday
He was reading a book
```

Tokenized:
```
i-am-tall
i-was-hungry-yesterday
he-was-reading-a-book
```

Add special tokens:
```
<sos>-i-am-tall-<eos>
<sos>-i-was-hungry-yesterday-<eos>
<sos>-he-was-reading-a-book-<eos>
```

Lookup indices:
```
1-100-503-321-2
1-100-312-254-2
1-781-312-431-120-678-2
```

Pad to same length:
```
1-100-503-321-2-0-0
1-100-312-254-2-0-0
1-781-312-431-120-678-2
```

### Model
#### Encoder
Encoding consists of:
1. Lookup the token indices in the token embedding table
2. Lookup the positions in the position embedding table (transformer has no sense of position otherwise)
3. Scale the token embedding and sum it up with the position embedding
4. Pass the unified embedding through multiple transformer encoder layers

#### Decoder
Decoding consists of:
1. Lookup the token indices in the token embedding table
2. Lookup the positions in the position embedding table (transformer has no sense of position otherwise)
3. Scale the token embedding and sum it up with the position embedding
4. Pass the unified embedding through multiple transformer encoder layers, making sure to apply the right mask
5. Pass the decoded output through a fully connected layer to get probability distribution over tokens in the vocabulary


### Training
1. Pass source batch through the encoder to get the memory
2. Pass target batch and memory through the decoder to get the outputs
3. Shift target batch one place to the right
4. Pass the outputs and shifted target batch through the cross entropy loss function
5. Backpropagte the loss

### Inference
1. Tokenize the sentence
2. Pass it through the encoder to get the memory
3. Start generating a sentence by passing a <sos> token and memory through the decoder
4. Sample a token from the decoder output
5. Pass <sos> token and generated token along with the memory through the decoder
6. Sample the next token from the decoder output
7. Continue adding sampled tokens and passing the sequence of tokens through the decoder until <eos> token is sampled from the output
