# Deep Learning Models
In this folder, I mainly write deep learning models (mostly in NLP) for quant
- PoNet: https://arxiv.org/abs/2110.02442v1.pdf
- BigBird: https://arxiv.org/pdf/2007.14062v2.pdf
- CosFormer: https://arxiv.org/abs/2202.08791.pdf
- GAU: 
- Informer: https://arxiv.org/abs/2012.07436v2.pdf
- Linformer: https://arxiv.org/abs/2006.04768v3.pdf
- LinearTransformer: https://arxiv.org/abs/2006.16236.pdf
- LocalAttention: https://arxiv.org/abs/2011.04006.pdf
- LongFormer: https://arxiv.org/abs/2004.05150v2.pdf
- MLPMixer: https://arxiv.org/pdf/2105.01601v4.pdf
- PerFormer: https://arxiv.org/abs/2009.14794v3.pdf
- ReFormer: https://arxiv.org/abs/2001.04451.pdf
- SinkhornTransformer: https://arxiv.org/abs/2002.11296.pdf
- SparseFormer: https://arxiv.org/abs/1904.10509.pdf
- Synthesizer: https://arxiv.org/abs/2005.00743v2.pdf
- Transformer: https://arxiv.org/abs/1706.03762.pdf
- XGBoost: https://arxiv.org/abs/1603.02754.pdf
- ResMLP: https://arxiv.org/pdf/2105.03404.pdf
- RepMLP: https://arxiv.org/abs/2105.01883.pdf

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

# Summary:
- PoNet: 
- BigBird: 
- CosFormer: 
- GAU:
    1. Gated Attention Unit
- GLU Transformer:
    1. Gated Linear Unit
    2. self-attention - layernorm - glu - layernorm
    3. use fully-connected layer to get mask and value
- Informer: 
- Linformer: 
    1. Change the sequence length, reduce the computational complexity
    2. short sequence length has lower bound, proved by Johnson–Lindenstrauss Lemma
    3. Normal transformer-liked architecture
- LinearTransformer: 
- LocalAttention: 
- LongFormer: 
- MLPMixer: 
    1. Do fully connected in two dimensions: first one is in features dimension, second one is in num_patches dimension
    2. For num_patches dimension, use pseudo fc (conv1d) instead
    3. Notice that, classic mlp we usually use decreasing hidden\_size during fc layers; However, in such MLP variants, we use expansion trick. For example: hs -> expansion\_ratio * hs -> hs
- ReFormer: 
- SinkhornTransformer: 
- SparseFormer: 
- Synthesizer: 
- Transformer: 
    1. Tradition, classical and fundamental Transformer model
    2. Encoder + Decoder architecture, only encoder part will be used.
    3. Multi-head attention is the vital technical part.
- PoNet:
    1. Differnet Pooling Layer
    2. Local Pooling; Global Pooling; Segment Pooling
- PoolFormer:
    1. Pooling Feature Extractor
    2. Normal transformer-liked architecture
- XGBoost: 
- ResMLP: 
    1. Very similar to ResMLP. 
    2. Use 2d-convolution to patchify the input image; ResMLP use reshape method (similar to ViT)
- RepMLP



