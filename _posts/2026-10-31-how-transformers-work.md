---
layout: post
title: "How Transformers Work"
date: 2026-10-31
categories: [AI, Transformers, Deep-Dive]
---

*How a Transformer moves a token from text to the next-word probabilities, and how those same blocks sit in an encoder–decoder pass: the source is read by the encoder alone, and the target prefix, including the fixed `<bos>` id, meets the encoder memory only at the decoder.*

---

## Contents

A token moves in this order. Each step says what enters and what leaves. The later sections are the workings of those steps.

1. **Source tokenization.** The text `the cat sat` enters. The tokenizer leaves integer ids, one per word, `[12, 47, 9]`. [Section 15](#15-encoder-and-decoder).
2. **Target prefix and `<bos>`.** The target side starts as its own input. `<bos>` is vocabulary id `1`, chosen once when the tokenizer is built and inserted by the program. It is not predicted. `le` is a target word already chosen, id `88`. The two ids `[1, 88]` leave this step. Training learns only the embedding row for id `1`. [The two inputs](#the-two-inputs).
3. **Embeddings.** Each id enters an embedding table and leaves as a row of width `d_model`. The source matrix is `3 × 4`. The target matrix is `2 × 4`. Those rows are trained vectors. [Tokenization and embedding](#tokenization-and-embedding).
4. **Position.** The same rows enter. Order is mixed in, either by adding a position vector or later by rotary position on queries and keys. The shapes do not change. [Where order comes from](#3-where-order-comes-from).
5. **Encoder self-attention.** Source vectors `3 × 4` enter. A norm rescales them. `Wq`, `Wk`, and `Wv` make queries, keys, and values. Scores are a `3 × 3` matrix, softmax turns each row into weights, and the value mix lets every source token read the whole source. `Wo`, dropout, and the residual add leave a new `3 × 4` stream. [Causal self-attention](#5-causal-self-attention) explains the units. The encoder uses them with no future mask. [Encoder block](#encoder-block).
6. **Encoder feed-forward, then memory `M`.** The `3 × 4` stream enters a norm. `W1` widens each row to `d_ff` detector scores, `σ` zeroes the negative scores, and `W2` shrinks the survivors back to `d_model`. The residual add leaves memory `M`, still `3 × 4`, one row for `the`, `cat`, and `sat`. [Feed-forward network](#6-the-feed-forward-network), [worked example](#8-a-worked-example), [residual and norm](#10-how-the-residual-add-and-the-norm-are-decided).
7. **Decoder causal self-attention.** Target vectors `2 × 4` enter. The same attention units run with the decoder’s own weights. The causal mask lets `<bos>` read only itself and lets `le` read `<bos>` and `le`. `M` does not enter this sublayer. A `2 × 4` stream leaves. [Decoder block](#decoder-block).
8. **Decoder cross-attention.** The decoder stream and memory `M` enter together. Queries come from the target rows. Keys and values come from `M`. Scores are `2 × 3`, one weight for `the`, `cat`, and `sat`. The value mix and `Wo` leave an update of shape `2 × 4`, added back onto the target stream. [Decoder block](#decoder-block).
9. **Decoder feed-forward.** The target stream `2 × 4` enters the decoder’s own `W1`, `σ`, and `W2`. The residual add leaves the decoder output, still `2 × 4`. Row 0 belongs to `<bos>`. Row 1 belongs to `le`. [What leaves the decoder](#what-leaves-the-decoder).
10. **Output.** The row for `le` enters the final norm, then the unembed matrix. Logits leave, one score per vocabulary word. Softmax turns them into probabilities, and the program picks the next target word, such as `chat`. The next pass enters the decoder as `<bos> le chat` with the same memory `M`. [After W2, through softmax](#9-after-w2-from-an-update-vector-to-probabilities).

---

## 1. The job of the model

A language-model Transformer reads a sequence of tokens and predicts what comes next. The sentence `the cat sat` goes in. A probability for every word in the vocabulary comes out, and a word is drawn from that distribution.

The same stack is used at every position during training. Position 0 predicts the word after `the`, position 1 predicts the word after `the cat`, and position 2 predicts the word after `the cat sat`. The loss compares each prediction with the token that actually followed.

```
 input:     the     cat     sat
 target:    cat     sat     ???
```

At generation time only the last position is sampled. That token is appended, and the model runs again.

---

## 2. The whole model

```
  "the cat sat"
        │
        ▼
    tokenize                text → integer ids, one per token
        │
        ▼
    embedding               each id → a vector of width d_model
        │                   position is mixed in here, or later inside attention
        ▼
   ┌─────────┐
   │ Block 1 │              attention, then the feed-forward network
   └─────────┘
        │
   ┌─────────┐
   │ Block 2 │
   └─────────┘
        │
       ...                   the same block, N times
        │
   ┌─────────┐
   │ Block N │
   └─────────┘
        │
        ▼
    final norm
        │
        ▼
    unembed                 each vector → one score per vocabulary word
        │
        ▼
    softmax                 scores → probabilities for the next token
```

The sequence length and the vector width stay the same through every block. Three tokens of width `d_model` are a matrix of shape `3 × d_model` on every arrow between blocks. Only the unembed step widens each row to the vocabulary size.

---

## 3. Where order comes from

A bag of token vectors has no left-to-right order. Attention would treat `cat sat the` the same as `the cat sat` unless position is added somewhere. Two usual places:

- A position vector is added to each token embedding at the bottom of the stack.
- Rotary position is applied to queries and keys inside attention. The match between token `i` and token `j` then depends on the distance `i − j`. Most current models do this.

---

## 4. One block

A full model is one block stacked many times. Inside a block, two sublayers run one after the other: causal self-attention, then a position-wise feed-forward network. Each sublayer is wrapped in a norm, a dropout, and a residual add. Data moves upward.

```
                 x
                 │
                 ├────────────────────────┐
                 ▼                        │
               Norm                       │
                 │                        │
                 ▼                        │  residual
      Causal multi-head attention         │
                 │                        │
                 ▼                        │
              Dropout                     │
                 │                        │
                 ▼                        │
               Add ◄──────────────────────┘
                 │
                 ├────────────────────────┐
                 ▼                        │
               Norm                       │
                 │                        │
                 ▼                        │  residual
        Position-wise feed-forward        │
                 │                        │
                 ▼                        │
              Dropout                     │
                 │                        │
                 ▼                        │
               Add ◄──────────────────────┘
                 │
                 ▼
              x  (to the next block)
```

```
x  ←  x + Dropout(Attention(Norm(x)))
x  ←  x + Dropout(FFN(Norm(x)))
```

The vertical line on the right is the residual. Each sublayer learns a change and adds it to the vector that was already there, so a token's representation is a running summary that the blocks edit. The norm in front keeps that summary at a steady scale before each edit. Dropout, used while training, zeros random entries so the model does not rely on any single unit.

This arrangement is called pre-norm: normalize, transform, drop out, add back to the un-normalized input. The two sublayers are in series. The feed-forward network reads the output of the attention add, so attention finishes before the feed-forward network starts.

Attention is the only place tokens exchange information. The feed-forward network then updates each token on its own.

---

## 5. Causal self-attention

The attention sublayer receives a matrix `X` of shape `T × d_model`: `T` tokens, each a vector of width `d_model`. The same `X` is multiplied by three learned matrices:

```
Q = X Wq
K = X Wk
V = X Wv
```

Row `i` of each result belongs to token `i`.

| Vector | Role |
|---|---|
| Query `q_i` | What token `i` is looking for |
| Key `k_j` | The label token `j` offers for matching |
| Value `v_j` | The content token `j` contributes if it is chosen |

`Wq`, `Wk`, and `Wv` are trained. They start random. Training moves them until a token's query lines up with the keys of the tokens it should read.

```
            token vectors X
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
      X Wq      X Wk      X Wv
        │         │         │
        Q         K         V
        │         │         │
        └────►  Q Kᵀ  ◄────┘
                  │
                  ▼
            ÷ √d_k
                  │
                  ▼
         causal mask: future scores → −∞
                  │
                  ▼
              softmax          each row sums to 1
                  │
                  ▼
               A × V           weighted mix of value vectors
                  │
                  ▼
               × Wo            joins the heads
                  │
                  ▼
            attention output
```

The dot product `q_i · k_j` is high when token `i` should read token `j`. Those dots form a `T × T` score matrix:

```
scores = (Q Kᵀ) / √d_k
```

`d_k` is the width of one query or key. Dividing by `√d_k` keeps the dots from growing just because the vectors are wide. Large dots would push softmax into a spike on a single token and starve the gradient.

**Causal mask.** Entry `(i, j)` is the attention from token `i` to token `j`. Every entry with `j > i` is set to `−∞` before softmax, so its weight becomes 0. Position `i` may use positions `1 … i` only. Later tokens stay hidden, which is what next-token prediction requires.

Illustrative scores for `the cat sat`. The softmax weights are rounded to two decimals.

```
        before the mask              after the mask              after softmax
        the    cat    sat            the    cat    sat           the    cat    sat
the      2      5      1        the   2     −∞     −∞        the  1.00  0     0
cat      0      3      4        cat   0      3     −∞        cat  0.05  0.95  0
sat      1      2      2        sat   1      2      2        sat  0.16  0.42  0.42
```

`the` can read only itself, so its weight on itself is 1. `cat` splits about 5% / 95% between `the` and `cat`. `sat` can read all three. In this example its new vector is about `0.16` of `v_the`, `0.42` of `v_cat`, and `0.42` of `v_sat`.

```
output row for sat  =  0.16·v_the  +  0.42·v_cat  +  0.42·v_sat
```

**Multi-head.** The same procedure runs several times in parallel. With `h` heads, each head has its own `Wq`, `Wk`, and `Wv`, and a narrower width, usually `d_model / h`. One head can learn to look at the previous word. Another can learn to look at a matching bracket, or at a subject several tokens back. The head outputs are concatenated back to width `d_model` and multiplied by one more matrix `Wo`, which mixes the heads.

That result leaves the attention box, passes through dropout, and is added to the residual:

```
X  ←  X + Dropout(Attention(Norm(X)))
```

---

## 6. The feed-forward network

After attention, every token has gathered context. The feed-forward box then processes each token alone, with the same two matrices at every position. "Position-wise" means the box never looks across tokens. Tokens mix only inside attention.

```
   token vector          width d_model
         │
         ▼
       × W1              widen to d_ff, often 4 × d_model
         │
         ▼
         σ               ReLU, GELU, or SiLU, applied to each entry
         │
         ▼
       × W2              shrink back to d_model
         │
         ▼
   updated vector
```

The original Transformer used ReLU and a bias on each linear map:

```
FFN(x) = max(0, x W1 + b1) W2 + b2
```

`W1` widens the vector, `b1` is added, ReLU keeps the positive part, `W2` shrinks the result, and `b2` is added on the way out. The same weights are shared across every token position. Only `x` changes.

Most current implementations drop both bias vectors. When the feed-forward network is the plain two-matrix kind, the layer is:

```
FFN(x) = σ(x W1) W2
```

`σ` stands for whichever activation the model uses: ReLU, GELU, or SiLU.

With `d_model = 512` and the usual width ratio, each token goes `512 → 2048 → 512`. The activation sits in that wider space, between the two multiplies.

### A gated feed-forward network

Many recent models use a gated variant, often SwiGLU, with three matrices and an elementwise product:

```
FFN(x) = (σ(x W_gate) ⊙ x W_up) W_down
```

The two-matrix formula above is the plain case. The gated form also drops the biases. The hidden width is often about `8/3 × d_model` rather than `4 × d_model`, so the parameter count stays in the same range as the two-matrix layer.

---

## 7. What W1 and W2 achieve

`W1` and `W2` turn one token vector into an update of the same width. `W1` scores a large bank of pattern detectors. `W2` writes the detectors that fired back into the residual stream.

One intermediate unit `i` is a stored pair: a detector `k_i` and an update direction `v_i`.

```
FFN(x) = Σ_i  σ(x · k_i) · v_i
```

`k_i` is column `i` of `W1`. `v_i` is row `i` of `W2`. The sum has `d_ff` terms, one per hidden unit.

```
x  (width d_model)
        │
        │  each detector: score = x · k_i
        ▼
     × W1                         d_model → d_ff
        │
        ▼
        σ                           a negative ReLU score contributes 0
        │
        │  output = score_1·v_1 + score_2·v_2 + ...
        ▼
     × W2                         d_ff → d_model
        │
        ▼
   update added to x
```

For a token vector `x`:

- `x · k_i` asks how strongly this token matches pattern `i`. A pattern might be "this token is a currency name," or "the attention step just marked this as the verb."
- `σ` turns that score on or off. With ReLU, a negative score contributes nothing, so most detectors stay silent for a given token.
- The detectors that survive add their vectors `v_i` into the result. `W2` is the stack of those update vectors.

`W1` reads the token. `W2` writes the consequence. The activation between them is what makes the layer nonlinear. The wide middle is a set of independent detectors because `σ` sits between the two matrices. A single square matrix `x W`, with no wide middle, can only rotate and rescale `x`.

### Why the middle is wide

The residual stream is a narrow channel of width `d_model`, shared by every block. The model needs far more detectors than that width can hold as separate axes. The hidden layer is a private workspace for this token and this block: often `d_ff = 4 × d_model`, so a 4096-wide stream gets 16384 detectors.

Each token turns on only a subset of them. The same matrices store many patterns. The dot product and `σ` select the few that apply to this token. Most of a Transformer's stored patterns sit in these pairs `(k_i, v_i)`.

### Why it shrinks again

The block's job is an addition to `x`:

```
x  ←  x + FFN(Norm(x))
```

That sum is legal only when `FFN(x)` has the same width as `x`. Attention in the next block also reads vectors of width `d_model`. The wide detector scores are useful inside this layer. `W2` folds them into the update that the rest of the model carries forward.

---

## 8. A worked example

The numbers below are small on purpose: `d_model = 2`, `d_ff = 3`, one token. Every entry of `W1` and `W2` is a stored weight. In a real model, training writes those weights. Here they are chosen so each product can be checked by hand.

```
x = [0.4,  1.0]
```

`W1` has one column per detector. Two rows, because `x` has width 2.

```
              k1      k2      k3
W1 = [      2.0    -1.0     0.0  ]     how much each detector uses x's 1st entry
     [      1.2     0.0     0.5  ]     how much each detector uses x's 2nd entry
```

`k1 = [2.0, 1.2]` means "look at both coordinates, and look harder at the first." `k2 = [−1.0, 0.0]` means "a large first coordinate is evidence against this pattern." These numbers are already in the matrix before `x` arrives. The multiply only reads them.

### W1 produces the scores

`x W1` dots `x` with each column:

```
score 1 = 0.4·(2.0)  +  1.0·(1.2)  =  0.8 + 1.2  =  2.0
score 2 = 0.4·(−1.0) +  1.0·(0.0)  = −0.4 + 0    = −0.4
score 3 = 0.4·(0.0)  +  1.0·(0.5)  =  0   + 0.5  =  0.5

raw scores:  [ 2.0,  −0.4,  0.5 ]
```

ReLU replaces the negative score with 0. Call the activated row `a`:

```
a = [ 2.0,  0,  0.5 ]
```

`W1` is finished once that score row exists. Detector 1 fired at strength 2, detector 2 stayed silent, detector 3 fired at strength 0.5.

### W2 multiplies the activated scores

`W2` has one row per detector and two columns, so the result has the same width as `x`.

```
                 output coord 1    output coord 2
detector 1       0.3               0.1            ← v1
W2 = detector 2  9.0               9.0            ← v2
     detector 3 −0.2               0.4            ← v3

shape of a:    1 × 3
shape of W2:   3 × 2
shape of a W2: 1 × 2
```

Each output entry is the score row dotted with one column of `W2`.

Output coordinate 1, dotted with column 1:

```
2.0·(0.3)  +  0·(9.0)  +  0.5·(−0.2)
= 0.6      +  0        +  (−0.1)
= 0.5
```

Output coordinate 2, dotted with column 2:

```
2.0·(0.1)  +  0·(9.0)  +  0.5·(0.4)
= 0.2      +  0        +  0.2
= 0.4
```

The update is `[0.5, 0.4]`. The same arithmetic as a sum of scaled rows:

```
2.0 · [ 0.3,  0.1]  =  [ 0.6,  0.2]     detector 1 fired at strength 2
0   · [ 9.0,  9.0]  =  [ 0,    0  ]     detector 2 was negative; ReLU zeroed it
0.5 · [−0.2,  0.4]  =  [−0.1,  0.2]     detector 3 fired at strength 0.5
                   sum  [ 0.5,  0.4]
```

Detector 2's row is `[9.0, 9.0]`, a large update, and it contributes nothing because its score is 0. `W2` only passes through detectors that survived `σ`.

### Two tokens, so the scores are a matrix

`W2` is applied to every token row on its own. Take a second token `x2 = [1.0, 0.0]`. Its scores are the first row of `W1`:

```
raw scores:  [ 2.0,  −1.0,  0.0 ]
after ReLU:  [ 2.0,   0,    0   ]
```

```
score matrix A          W2                        updates
[ 2.0   0    0.5 ]      [  0.3   0.1 ]            [ 0.5   0.4 ]     token 1
[ 2.0   0    0   ]  ×   [  9.0   9.0 ]      =     [ 0.6   0.2 ]     token 2
                        [ −0.2   0.4 ]

token 2:  2.0·[0.3, 0.1] + 0·[9.0, 9.0] + 0·[−0.2, 0.4] = [0.6, 0.2]
```

Both output rows have width 2, the same as the token vectors they get added to. In a real layer the score matrix is `T × d_ff` and `W2` is `d_ff × d_model`. Each of the `T` rows is combined exactly this way.

---

## 9. After W2: from an update vector to probabilities

The row that leaves `W2` is still a token vector. In the example it is `[0.5, 0.4]`. That row is an edit to the residual stream. It is not a probability, and it is not yet a score for a vocabulary word. Probabilities appear once, at the very top of the model, after every block has finished.

### The update is added, then the stack continues

Inside the block the feed-forward network runs on the normalized token, and its output is dropped out and added back:

```
update = σ(Norm(x) W1) W2          shape 1 × d_model, here [0.5, 0.4]
x      ← x + Dropout(update)
```

Using the token that entered this picture, `x = [0.4, 1.0]`, and skipping dropout for the arithmetic:

```
[0.4, 1.0] + [0.5, 0.4] = [0.9, 1.4]
```

`[0.9, 1.4]` is the token's new representation. If another block sits above this one, that vector goes into the next norm, the next attention mix, and another feed-forward update. Each block adds its own edit. The width stays `d_model` the whole way.

```
[0.5, 0.4]  from W2
      │
      ▼
 residual add ──► [0.9, 1.4]
      │
      ▼
 next block, and the block after that
      │
      ▼
 final norm ──► h     still width d_model
      │
      ▼
 × W_U ──► logits      one score per vocabulary word
      │
      ▼
 softmax ──► probabilities that sum to 1
```

### The unembed turns the vector into scores

After the last block, a final norm produces a vector `h`, still of width `d_model`. A learned matrix `W_U`, shaped `d_model × vocab_size`, scores every word in the vocabulary:

```
logits = h W_U
```

Each column of `W_U` is the direction for one word. The dot product of `h` with that column is the score for that word. A large positive score means the model currently favors that word. Scores are ordinary real numbers: they can be negative, and they do not sum to 1. Many models set `W_U` equal to the transpose of the token-embedding matrix, so the same vectors that turn ids into token vectors also turn the final vector back into word scores.

Suppose the vocabulary is three words that could follow `the cat sat`, and the final vector is `h = [1.0, 0.5]`.

```
              on     down    still
W_U = [     1.0     0.0      1.0  ]
      [     2.0     2.0     −2.0  ]
```

```
logit(on)    = 1.0·(1.0) + 0.5·(2.0)  = 2.0
logit(down)  = 1.0·(0.0) + 0.5·(2.0)  = 1.0
logit(still) = 1.0·(1.0) + 0.5·(−2.0) = 0.0

logits = [2.0,  1.0,  0.0]
```

Shape check: `h` is `1 × 2`, `W_U` is `2 × 3`, logits are `1 × 3`. With a real vocabulary the last dimension is tens of thousands, one entry per word.

### Softmax turns scores into probabilities

Softmax exponentiates every score and divides by the total, so the results are positive and sum to 1:

```
P(word j) = exp(logit_j) / Σ_k exp(logit_k)
```

```
exp(2.0) = 7.389
exp(1.0) = 2.718
exp(0.0) = 1.000
sum      = 11.107

P(on)    = 7.389 / 11.107 = 0.665
P(down)  = 2.718 / 11.107 = 0.245
P(still) = 1.000 / 11.107 = 0.090
                            -----
                            1.000
```

`on` had the largest logit, so it receives the largest probability. The exponential stretches the gaps: a logit of 2 against a logit of 0 becomes a probability of 0.665 against 0.090.

Generation draws the next token from this distribution. Taking the maximum always picks `on`. Sampling can still pick `down` or `still`, in proportion to those probabilities. Dividing the logits by a temperature before the softmax changes the shape of the distribution: a temperature below 1 makes the largest probability sharper, and a temperature above 1 spreads probability toward the other words.

During training, every position has its own row of probabilities. The loss is the cross-entropy against the true next token. If the true next word is `down`, the loss at this position is the negative log of its probability:

```
loss = −log(0.245) ≈ 1.41
```

A higher probability on the true word makes this number smaller, which is what the gradient steps in section 11 are chasing.

The softmax inside attention is a different distribution. There, each row is a mix over earlier positions in the sequence, and it is used to combine value vectors. The softmax here is a mix over the vocabulary, and it is the model's prediction of the next word.

---

## 10. How the residual add and the norm are decided

The residual add and the norm are fixed formulas. Training does not choose whether to add, or how the rescaling is computed. It only learns a scale vector inside the norm. Both formulas work element by element, so the width stays `d_model`.

### The residual add

The add is elementwise. The vector that skipped around the feed-forward network is added to the update that came out of `W2`:

```
x_new = x + update
[0.4, 1.0] + [0.5, 0.4] = [0.9, 1.4]
```

Nothing is learned in that line. There is no weight matrix on the add. The block is built so the sublayer output has the same width as `x`, and those two vectors are summed. The same add happens after attention, with that sublayer's own update:

```
x  =  x + attention update
x  =  x + feed-forward update      ← the [0.9, 1.4] step
```

Each later block repeats those two adds with its own updates. The running vector is whatever the previous add produced.

In the block, the feed-forward network reads the normalized `x`. The skip still carries the raw `x`:

```
x ─────────────────────────────┐
│                              │
▼                              │
Norm                           │  this is the vector W1 actually sees
│                              │
W1 → σ → W2 → update           │
│                              │
▼                              │
x + update  ◄──────────────────┘
```

The worked `W1` / `W2` numbers in section 8 were applied straight to `[0.4, 1.0]` so the multiply stayed easy to check. In the block, that multiply runs on `Norm(x)`, and `[0.4, 1.0]` is the vector added back afterward.

### The norm

The norm recomputes a scale from the current vector, then multiplies by a learned vector `γ` of width `d_model`. `γ` starts as all ones and moves during training the same way `W1` and `W2` do. The rescaling rule itself is part of the architecture.

Most current models use RMSNorm. For `x = [0.9, 1.4]` after the add, with `γ = [1, 1]`:

```
RMS = sqrt( (0.9² + 1.4²) / 2 ) = sqrt(1.385) = 1.177

h = γ ⊙ x / RMS
  = [0.9 / 1.177,  1.4 / 1.177]
  = [0.765,  1.190]
```

A tiny constant `ε` is added inside the square root so a zero vector is never divided by zero. The designer picks `ε`, often `10⁻⁵` or `10⁻⁶`. It is not trained.

LayerNorm, the older choice, also subtracts the mean and has a second learned vector `β` added at the end:

```
mean = (0.9 + 1.4) / 2 = 1.15
std  = 0.25
(x − mean) / std = [−1,  1]

h = γ ⊙ (x − mean) / std + β
```

With `γ = [1, 1]` and `β = [0, 0]`, that `h` is `[−1, 1]`. Dropping `β` is the same kind of choice as dropping `b1` and `b2`.

Each block owns its own `γ`. Block 3 does not reuse block 1's scale. The formula is identical in every block.

### Where `h` comes from

The final norm is one extra RMSNorm, or LayerNorm, after the last block, with its own `γ`. There is no residual wrapped around it. Its output is `h`, still width `d_model`, and that is the vector multiplied by `W_U`:

```
[0.9, 1.4]     last residual add
     │
     ▼
 final norm    h = γ ⊙ x / RMS(x)     →  [0.765, 1.190] when γ is all ones
     │
     ▼
 h W_U         logits, one score per vocabulary word
     │
     ▼
 softmax
```

A plain sum produces the next residual stream. Every block, and the final layer, rescale that stream with the norm formula. Only `γ` inside each norm is a learned parameter.

---

## 11. Where the weight values come from

The entries of `W1` and of `W2` are stored parameters. Training writes them, over the whole corpus, before any new token is seen. A forward pass reads the matrices. It does not invent their entries from the current token.

1. **Start random.** Every entry of `W1` and `W2` is drawn from a small random distribution. At this point no column means "currency name."
2. **Predict.** A batch of real text is run through the model. Each position produces a probability for the next token.
3. **Measure the miss.** Cross-entropy compares that distribution with the token that actually came next. A wrong prediction makes a large loss.
4. **Assign blame to each entry.** Backpropagation computes, for every number in both matrices, whether nudging it up or down would have made the loss smaller. That direction is the gradient.
5. **Rewrite the entry.** The optimizer steps every weight a small amount: `W ← W − learning_rate · gradient`.

One entry of a detector might move like this:

```
step            entry of k1        what has happened
random start    0.08               no meaning yet
mid training    0.4                raising it made some next-token predictions better
late training   2.0                this direction now fires for a useful pattern
```

The same update hits every entry of `W1` and every entry of `W2`. A column of `W1` becomes a detector because the loss kept rewarding that direction: tokens where the pattern matters push `x · k_i` upward, and tokens where the pattern should stay silent push it downward. A row of `W2` becomes the update to add when that detector fires. After training, both matrices are saved with the model and reused unchanged.

---

## 12. How the shapes are decided

Two numbers fix both shapes. The model designer picks them before training, as part of the architecture. Training fills the entries. It never changes the shapes.

| Symbol | Meaning | Typical value |
|---|---|---|
| `d_model` | Width of each token vector, used by embeddings, attention, and the residual add | 512 in the original Transformer |
| `d_ff` | Number of detectors in the feed-forward layer | `4 × d_model`, so 2048 |

Matrix multiplication allows `(m × n) (n × p)`. The touching dimensions are equal, and the outer ones survive. That rule forces every dimension once `d_model` and `d_ff` are chosen.

```
x            W1                 a = σ(x W1)          W2                  update
[ *  * ]  ×  [ *  *  * ]    =   [ *  *  * ]      ×   [ *  * ]        =   [ *  * ]
  1 × 2       [ *  *  * ]         1 × 3              [ *  * ]            1 × 2
               2 × 3                                  [ *  * ]
                                                      3 × 2
```

`W1` sits between `x` and the score row `a`:

- Its row count equals the width of `x`, so `d_model`. Each row holds one coordinate of every detector.
- Its column count is the number of detectors, `d_ff`. That column count becomes the length of `a`: one score per detector.

`W2` sits between `a` and the update:

- Its row count equals the length of `a`, so `d_ff`. Row `i` is the update vector `v_i` for detector `i`.
- Its column count equals the width of `x`, so `d_model`. The update must match `x` so the residual add is legal.

```
W1:   d_model × d_ff
a:    1 × d_ff
W2:   d_ff × d_model
```

At the original Transformer's widths:

```
x          1 × 512
W1       512 × 2048
a          1 × 2048        one score per detector
W2      2048 × 512
update     1 × 512         same shape as x
```

A longer sentence does not change `W1` or `W2`. For `T` tokens, `a` is `T × d_ff` and the update is `T × d_model`. The weight matrices stay `d_model × d_ff` and `d_ff × d_model`, shared by every token.

---

## 13. One token walking through

Start with `sat` in `the cat sat`.

1. Embedding turns `sat` into a vector. Position information says it is the third token.
2. In block 1, its query is compared with the keys of `the`, `cat`, and `sat`. The causal mask hides anything after `sat`. Softmax picks a mix of those three value vectors. The feed-forward network then scores its detectors with `W1`, keeps the ones that fire, and uses `W2` to write an update of width `d_model` back onto the token.
3. Blocks 2 through `N` repeat the edit. Later blocks read representations that earlier blocks have already filled with context.
4. The final norm rescales that vector, as in section 10, and the unembed matrix turns it into one score per vocabulary word. Softmax turns those scores into probabilities that sum to 1, as in the worked example of section 9, and the model samples the next word from that distribution.

Stacking these blocks is the language-model path: attention moves information between tokens, the feed-forward network updates each token, and the residual path carries the representation up to the next-token prediction. Section 15 uses the same units in an encoder–decoder pass, where a source sentence and a target prefix are separate inputs.

---

## 14. Why the bias terms go away

The original layer adds a bias after each multiply: `b1` in the wide space, `b2` in the stream. Current models leave both out. Two reasons, the same idea as the move from LayerNorm to RMSNorm.

**Memory.** A bias is one extra vector per linear layer (`d_ff` for `b1`, `d_model` for `b2`), stored and added on every forward and backward pass. Next to `W1` and `W2` the count is small, but the choice is repeated in every block, and the same choice is usually applied to the attention projections and to the norm. RMSNorm makes the matching cut inside the norm: LayerNorm has a learned scale and a learned shift, and RMSNorm keeps the scale and drops the shift. Both moves delete an additive offset the model can do without.

**Optimization stability.** At large scale, training is steadier without those offsets. A bias right before a norm is partly redundant, because the norm recenters or rescales the vector and can relearn a shift with its own parameters. Biases on the output side also sit on the residual stream, where a constant push can fight the norm and the residual add. Large models trained this way keep quality while the run stays more stable.

---

## 15. Encoder and decoder

A language model is a stack of decoder blocks with no encoder beside it. A translation model has both. The source sentence is read by the encoder alone. The translation written so far is a second input that meets the encoder’s memory only at the decoder.

This pass uses source `the cat sat` and target prefix `<bos> le`, with `d_model = 4`, 2 heads so `d_k = 2`, and `d_ff = 8`. The attention, feed-forward, residual add, and norm inside each box are the units from sections 4 through 10.

```
SOURCE  (the sentence being read)          TARGET PREFIX  (the translation so far)
"the cat sat"                               "<bos>  le"
      │                                          │
      │  the, cat, sat are the source words      │  <bos> = fixed start id, inserted by the program
      ▼                                          │  le    = first target word, already chosen
ids [12, 47, 9]                                  ▼
      │                                     ids [1, 88]
      ▼                                          │
source vectors  3 × 4                            ▼
      │                                     target vectors  2 × 4
      ▼                                          │
┌─────────────┐                                  │
│   ENCODER   │  one input only                  │
│  attention  │                                  │
│  then FFN   │                                  │
└─────────────┘                                  │
      │                                          │
      ▼                                          │
memory M  3 × 4                                  │
one row for the, cat, sat                        │
      │                                          │
      │         both arrive together             │
      └──────────────┐    ┌──────────────────────┘
                     ▼    ▼
              ┌─────────────────┐
              │     DECODER     │  two inputs at once
              │ self-attention  │  • M, the source memory
              │ cross-attention │  • <bos> le, the target so far
              │ then FFN        │
              └─────────────────┘
                     │
                     ▼
              decoder output  2 × 4
              row 0 = <bos>,  row 1 = le
                     │
                     ▼
              the row for le becomes probabilities
              for the next target word, e.g. chat
```

### The two inputs

The source is the sentence the encoder reads: `the`, `cat`, and `sat`. Each word becomes one row of width 4. The encoder rewrites those three rows and leaves memory `M`, still `3 × 4`. Row 0 remains `the`, row 1 `cat`, row 2 `sat`.

The target prefix is the translation so far. It is not an input to the encoder.

* `<bos>` means beginning of sequence. It is a reserved vocabulary slot, chosen once when the tokenizer is built. In this example that slot is id `1`. Any other unused integer would have worked. After the choice, `1` always means “start” and is never reassigned per sentence. The program inserts it: the training loader writes id `1` in front of each reference translation, and the generator writes id `1` as the first decoder input before any word exists. The model is not asked to predict `<bos>`. Training learns only the embedding row for id `1`. That row starts random, like the row for `le`, and gradient updates move its `d_model` numbers.
* `le` is a real target word already chosen. During generation, the previous pass produced it. During training, it is copied from the reference translation. Its id, `88`, is the ordinary vocabulary id for that word, and its embedding row is learned the same way.

`<bos>` and `le` are embedded on their own into a `2 × 4` matrix. That work can happen while the encoder is still running, because it does not use the source. The decoder block is the first place the two streams meet, and it runs only once `M` exists. Later target words do not cause another encoder pass. `M` stays fixed.

Generation builds the prefix one word at a time. The first input contains only the fixed start id:

```
pass 1 input:   <bos>                  output word: le
pass 2 input:   <bos>  le              output word: chat
pass 3 input:   <bos>  le  chat        output word: ...
```

The diagram is pass 2. Id `1` was inserted by the program. Id `88` (`le`) is the word pass 1 already produced.

### Tokenization and embedding

| Step | Enters | Leaves |
|---|---|---|
| Source tokenizer | `the cat sat` | ids `[12, 47, 9]` |
| Target tokenizer | `<bos> le` | ids `[1, 88]`. Id `1` is the reserved start slot, not a word looked up from the sentence |
| Source embedding | those 3 ids | source vectors, `3 × 4`. Each row is a trained vector |
| Target embedding | those 2 ids | target vectors, `2 × 4`. The row for id `1` is the trained `<bos>` vector |
| Position | each row | the same shapes. A position vector is added here, or rotary position is applied later inside attention |

### Encoder block

One input, `X`, shape `3 × 4`. One output, memory `M`, shape `3 × 4`. Attention runs first, then the feed-forward block. A stack of encoder blocks repeats these units. The shapes stay the same.

Self-attention here lets every source token read every source token.

| Unit | Enters | Leaves |
|---|---|---|
| Norm | `X`, `3 × 4` | rescaled `X`, `3 × 4`. The skip keeps the original `X` |
| `Wq`, `Wk`, `Wv` | `3 × 4` | `Q`, `K`, `V`, each `3 × 4`, split into 2 heads of `3 × 2` |
| Scores | `Q` and `K` per head | `3 × 3`. Entry `(i, j)` is how much source token `i` wants source token `j` |
| Mask | `3 × 3` | `3 × 3`. No future token is hidden. A padding position, if the batch has one, is set to −∞ |
| Softmax | `3 × 3` | `3 × 3`. Each row sums to 1 |
| Value mix | weights `3 × 3` and `V` `3 × 2` | `3 × 2` per head. Each source token is a mix of all three value rows |
| Concatenate and `Wo` | two heads | one update, `3 × 4` |
| Dropout | `3 × 4` | `3 × 4`, with random zeros during training |
| Residual add | original `X` plus the update | new `X`, `3 × 4` |

The feed-forward sublayer then updates each source token on its own.

| Unit | Enters | Leaves |
|---|---|---|
| Norm | `X`, `3 × 4` | `3 × 4`. The skip keeps the pre-norm `X` |
| `W1` | `3 × 4` | detector scores `3 × 8` |
| `σ` | `3 × 8` | `3 × 8`. Negative scores become 0 |
| `W2` | `3 × 8` | update `3 × 4` |
| Dropout | `3 × 4` | `3 × 4` |
| Residual add | pre-norm `X` plus the update | `M`, `3 × 4` |

`M` leaves the encoder and is held for every decoder pass.

### Decoder block

Two inputs arrive together: memory `M` at `3 × 4`, and target vectors `Y` at `2 × 4`. Three sublayers run in order. The output is still `2 × 4`.

Causal self-attention lets the target tokens read each other. `M` is not used in this sublayer.

| Unit | Enters | Leaves |
|---|---|---|
| Norm | `Y`, `2 × 4` | `2 × 4` |
| `Wq`, `Wk`, `Wv` | `2 × 4` | `Q`, `K`, `V` as 2 heads of `2 × 2`. These weights belong to the decoder |
| Scores | `Q`, `K` | `2 × 2` |
| Causal mask | `2 × 2` | `2 × 2`. `<bos>` can read only itself. `le` can read `<bos>` and `le`. The other entry is −∞ |
| Softmax | masked scores | `2 × 2`, rows sum to 1 |
| Value mix, `Wo`, dropout | head values | update `2 × 4` |
| Residual add | original `Y` plus the update | `Y`, `2 × 4` |

Cross-attention is the unit that reads the encoder. Queries come from the decoder. Keys and values come from `M`.

| Unit | Enters | Leaves |
|---|---|---|
| Norm | decoder `Y`, `2 × 4` | `2 × 4` |
| `Wq` | decoder `2 × 4` | `Q`, heads of `2 × 2` |
| `Wk`, `Wv` | memory `M`, `3 × 4` | `K` and `V`, heads of `3 × 2` |
| Scores | `Q` and `K` | `2 × 3`. Each target token has one score for `the`, `cat`, and `sat` |
| Mask | `2 × 3` | `2 × 3`. All source positions stay visible |
| Softmax | `2 × 3` | `2 × 3`, each target row sums to 1 |
| Value mix | weights `2 × 3` and encoder `V` `3 × 2` | `2 × 2` per head. `le` is a mix of the three source values |
| `Wo`, dropout | concatenated heads | update `2 × 4` |
| Residual add | decoder `Y` plus the update | `Y`, `2 × 4` |

The feed-forward sublayer uses the same units as the encoder’s, with the decoder’s own `W1`, `W2`, and norm scale.

| Unit | Enters | Leaves |
|---|---|---|
| Norm | `Y`, `2 × 4` | `2 × 4` |
| `W1` | `2 × 4` | `2 × 8` |
| `σ` | `2 × 8` | `2 × 8` |
| `W2` | `2 × 8` | `2 × 4` |
| Dropout and residual add | the update plus the pre-norm `Y` | decoder output, `2 × 4` |

### What leaves the decoder

The block’s output is `2 × 4`. Row 0 belongs to `<bos>`, the trained vector for the fixed id `1`, rewritten by the block. Row 1 belongs to `le`. The next decoder block, if there is one, takes this matrix as its new `Y` and reads the same `M`.

After the last decoder block, the row for `le` is normalized and multiplied by the unembed matrix, as in section 9. That produces one score per vocabulary word. Softmax turns the scores into probabilities, and the program picks the next target word, such as `chat`. The following pass feeds `<bos> le chat` into the decoder with the same memory `M`. The start id is still `1`.
