---
layout: post
title: "How Embeddings Works"
date: 2026-11-01
categories: [AI, Transformers, Deep-Dive]
---

*How a token id becomes the vector a block can read: the embedding table, the lookup, position, and the matrix that enters the encoder or the decoder.*

---

## Contents

An id moves in this order. Each step says what enters and what leaves.

1. **An id arrives.** The tokenizer has already left integers. `the cat sat` is `[12, 47, 9]`. `<bos> le` is `[1, 88]`.
2. **The embedding table.** A matrix `E` of shape `vocabulary size × d_model` holds one row per id. In this post `d_model = 4`.
3. **Initialization.** Every entry of `E` starts as a small random number. No row means “cat” yet.
4. **Lookup.** An id enters. The row at that index leaves. Id `47` leaves the four numbers stored for `cat`.
5. **The token matrix.** The rows are stacked in order. Three source ids leave a `3 × 4` matrix. Two target ids leave a `2 × 4` matrix.
6. **Position.** Those rows enter. A position vector is added, or a rotation is kept to be applied later to queries and keys. The shape stays `T × 4`.
7. **Into a block.** The source matrix enters the encoder only. The target matrix enters the decoder only.
8. **Training.** The loss sends a gradient back to each looked-up row, and the optimizer nudges that row. Rows absent from the batch stay put.
9. **Scoring words.** When the input and output weights are tied, the final vector is dotted with every row of `E`. Each dot product is one logit.

---

## 1. What an embedding is

An embedding is a learned vector stored under an integer id. The tokenizer has already chosen the id. The embedding table stores `d_model` numbers for that id.

```
id 47  ──lookup──►  row 47 of E  ──►  [0.5, -0.2, 0.3, 0.1]
```

The table in this post has one row per vocabulary id and 4 columns. Only the rows this pass uses are shown:

```
id    piece     row of E
 1    <bos>     [0.1,  0.1,  0.1,  0.1]
 9    sat       [0.0,  0.6,  0.2, -0.1]
12    the       [0.2,  0.1,  0.0,  0.4]
47    cat       [0.5, -0.2,  0.3,  0.1]
88    le        [0.4,  0.0,  0.2,  0.3]
```

`E` has shape `V × 4`, where `V` is the vocabulary size. A model with a vocabulary of 32,000 and `d_model = 512` stores a matrix of shape `32000 × 512`. The toy table is the same object with both numbers shrunk.

The four numbers are not assigned by a person. After training, a direction in this space is useful because dot products with it made the next-token loss smaller. Two ids that the loss treats in a similar way tend to end up with rows that point in similar directions.

---

## 2. How the lookup is implemented

The forward pass reads a row by index. It does not build a one-hot vector in memory:

```
function embed(ids, E):
    rows ← empty list
    for id in ids:
        append E[id] to rows
    return stack(rows)          shape T × d_model
```

That read is the same result as a one-hot multiply. The one-hot vector for `cat` has a 1 in slot 47 and a 0 in every other slot. Multiplying it by `E` selects row 47:

```
one_hot(47) × E  =  E[47]  =  [0.5, -0.2, 0.3, 0.1]
```

The implementation allocates `E` as a parameter matrix, fills it once at initialization, and gathers rows. Gathering is the whole forward computation of the embedding layer.

A typical initialization draws each entry from a small normal distribution, with standard deviation about `0.02`, or from a uniform range scaled by `1 / sqrt(d_model)`. The original Transformer then multiplies the looked-up token rows by `sqrt(d_model)` before adding position. With `d_model = 4` that scale is `2`. Many later models leave the rows unscaled and rely on the norm inside the first block.

---

## 3. From ids to the source and target matrices

Source ids and target ids are looked up separately and stacked in order.

```
source ids [12, 47, 9]

E[12]  the    [0.2,  0.1,  0.0,  0.4]
E[47]  cat    [0.5, -0.2,  0.3,  0.1]
E[9]   sat    [0.0,  0.6,  0.2, -0.1]

source matrix X    shape 3 × 4
```

```
target ids [1, 88]

E[1]   <bos>  [0.1,  0.1,  0.1,  0.1]
E[88]  le     [0.4,  0.0,  0.2,  0.3]

target matrix Y    shape 2 × 4
```

Id `1` is the reserved `<bos>` slot. The program inserts `1`. Row `E[1]` is an ordinary learned row. This pass reads it. It does not create it.

If the source language and the target language share one vocabulary, both lookups use the same `E`. If they have different vocabularies, the encoder has one table and the decoder has another. The function is the same: an id goes in, a row comes out.

A padding id also has a row. The attention mask and the loss ignore pad positions, so that row is read and then left out of the prediction.

---

## 4. Position

The rows above do not yet say which word came first. Position is added in one of three ways.

### Learned position vectors

A second table `P` has one row per position. Row 0 is added to the first token, row 1 to the second, and so on.

```
P[0]  [0.01, 0.00, 0.02, 0.00]
P[1]  [0.02, 0.01, 0.00, 0.01]
P[2]  [0.00, 0.02, 0.01, 0.02]
```

```
the  [0.2,  0.1,  0.0,  0.4] + P[0]  =  [0.21,  0.10,  0.02,  0.40]
cat  [0.5, -0.2,  0.3,  0.1] + P[1]  =  [0.52, -0.19,  0.30,  0.11]
sat  [0.0,  0.6,  0.2, -0.1] + P[2]  =  [0.00,  0.62,  0.21, -0.08]
```

`P` is initialized randomly and trained the same way as `E`. Its shape is `maximum length × d_model`. A model that only allocated 512 position rows has no row to add at position 512.

### Fixed sinusoidal position

The original Transformer fills `P` with a formula. The rows are not gradient parameters. For position `pos` and dimension pair `i`:

```
P(pos, 2i)     = sin( pos / 10000^(2i / d_model) )
P(pos, 2i + 1) = cos( pos / 10000^(2i / d_model) )
```

With `d_model = 4`:

```
pos 0  →  [sin 0,  cos 0,  sin 0,      cos 0    ]  =  [0,     1,      0,     1    ]
pos 1  →  [sin 1,  cos 1,  sin 0.01,   cos 0.01 ]  =  [0.841, 0.540,  0.010, 0.999]
```

These rows are added to the token rows in the same way as learned `P`.

### Rotary position

No position vector is added to the token rows. Inside attention, each query and key is rotated by an angle that depends on its position. The matrix `E` still only supplies the token rows. The `3 × 4` matrix that enters the block is the stacked token rows.

---

## 5. What enters the encoder and the decoder

After lookup and position, the two matrices follow separate paths. They meet at the decoder.

```
X  3 × 4     the, cat, sat          enters the encoder only
                 │
                 ▼
             encoder
                 │
                 ▼
             memory M

Y  2 × 4     <bos>, le              enters the decoder only
                 │
                 └──────────►  decoder  ◄──── M
```

`X` is the encoder’s only input. `Y` does not join it. The decoder block is the first place `Y` and `M` are read together. A language-model stack has no encoder, so the single matrix, token rows plus position, enters the first decoder block directly.

Dropout is often applied to these rows before the first block. Each kept entry is scaled so the expected size stays put. The shape does not change.

---

## 6. How training writes a row

`E` starts random. A training step looks up only the ids in the batch. After the loss is computed, backpropagation produces a gradient the same shape as each used row. The optimizer steps those rows:

```
E[id]  ←  E[id] − learning_rate × gradient[id]
```

On a batch that contains `cat` and does not contain `sat`, row `47` moves and row `9` stays still. Over the corpus, an id that appears often gets many such steps. The row for `<bos>`, id `1`, is updated whenever a target sequence is trained, because the program puts `1` on every target input.

The gradient of a row says which way to nudge those four numbers so the final next-token probabilities put more mass on the true next id. The embedding update is the same kind of step that moves `W1` and `W2`.

---

## 7. Tying the table to the output

The unembed matrix maps a final vector of width `d_model` to one score per vocabulary id. Its shape is `d_model × V`. The embedding table is `V × d_model`. Those are transposes of each other, so one stored matrix can serve both jobs.

The final vector for a token is dotted with every row of `E`. With `h` of width 4:

```
h = [0.2, -0.1, 0.4, 0.0]

score(the) = h · E[12] = 0.2·0.2 + (-0.1)·0.1 + 0.4·0.0 + 0.0·0.4 = 0.03
score(cat) = h · E[47] = 0.2·0.5 + (-0.1)·(-0.2) + 0.4·0.3 + 0.0·0.1 = 0.24
```

`cat` scores higher because `h` lines up more with row `47` than with row `12`. Softmax then turns the full list of `V` scores into probabilities. When the weights are tied, a training step updates a row from both sides: from being the input vector of that id, and from being the output direction that scores that id.
