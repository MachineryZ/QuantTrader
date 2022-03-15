# Deep Learning Models
In this folder, I mainly write deep learning models (mostly in NLP) for quant
- PoNet
- BigBird
- CosFormer
- Informer
- LinearTransformer
- LocalAttention
- LongFormer
- MLPMixer
- PerFormer
- PoNet
- ReFormer.py
- SinkhronTransformer
- SparseFormer
- Synthesizer
- Transformer: https://arxiv.org/abs/1706.03762

# Data
For testing, or for debugging, we will generate some pseudo data for our NN.
1. Sequence data:
    1. x in (batch\_size, seq\_len, feature\_size), factors
    2. y in (batch\_size, 1), targets, such as returns, volatiliy
2. Graph data:
    1. Category-based data, which means every data will be tagged a label in every
    relationship
        1. x in (batch\_size, seq\_len, feature\_size), factors
        2. y in (batch\_size, 1), targets, such as returns, volatiliy
        3. adj in (num\_of\_graphs, batch\_size, label)
    2. Relation-based data, which means the relation ship of each node will be represented
    as weight. (maybe with self loop, multiple graphs)
        1. x in (batch\_size, seq\_len, feature\_size), factors
        2. y in (batch\_size, 1), targets, such as returns, volatiliy
        3. adj in (num\_of\_graphs, batch\_size, batch\_size, weight)


