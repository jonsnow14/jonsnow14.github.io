---
layout: post
title: "How Tokenization Works"
date: 2026-11-01
categories: [AI, Transformers, Deep-Dive]
---

*How text becomes the integer ids a model can embed. Normalization first rewrites the input into one standard spelling, the canonical string. Everything after that step splits that spelling into pieces and looks the pieces up.*

---

## Contents

Text moves in this order. Each step says what enters and what leaves.

1. **Raw text.** The original string enters, with whatever case, spacing, and Unicode characters the user typed.
2. **Normalization.** That string enters a fixed set of rewrite rules. One standard spelling leaves. That spelling is the canonical string.
3. **Pretokenization.** The canonical string enters. A splitter leaves chunks that later merges may not cross.
4. **Symbols.** Each chunk enters. The learner leaves base symbols: words, characters, or bytes.
5. **Word tokenization.** A chunk enters a word vocabulary. A known word leaves its id. An unknown word leaves `<unk>`.
6. **Character tokenization.** A chunk enters one character at a time. Each character leaves its own id.
7. **Byte-pair encoding.** Characters enter. Training stores an ordered list of merges. Encoding applies those merges and leaves subword pieces, then ids.
8. **WordPiece.** Characters enter. Training keeps the pieces that raise the likelihood of the text. Encoding takes the longest match and leaves pieces, then ids.
9. **Unigram and SentencePiece.** A large seed vocabulary enters. Training drops the pieces that matter least. Encoding leaves the most probable segmentation, then ids.
10. **Byte-level encoding.** UTF-8 bytes enter. Merges build larger pieces on top of the 256 byte values. Every string leaves ids.
11. **Special ids.** The program adds reserved ids such as `<bos>`, `<eos>`, and `<pad>`.
12. **Ids to the embedding table.** The integer sequence leaves the tokenizer. Each id is read as one trained row.

---

## 1. The path from text to a vector

```
raw text
"The  Cat  Sat"                         "Café"  (e + combining accent)
"Ｔｈｅ  cat"  (fullwidth letters)       "ﬁcat"
        │                                       │
        ▼                                       ▼
normalize to one spelling
  lowercase              The → the, Cat → cat, Sat → sat
  collapse spaces        "The  Cat" → "The Cat"
  Unicode NFKC           Ｔ → T, ﬁ → fi
                         e + combining accent → é
  strip accents          only if this tokenizer chose that rule: é → e
        │                                       │
        ▼                                       ▼
canonical string
"the cat sat"                           "ficat"   (or "fiécat" if accents are kept)
the single form stored for the next step
        │
        ▼
pretokenize        chunks later merges may not cross
  split on spaces and punctuation, or mark each space with ▁
        │
        ▼
"the"   "cat"   "sat"
        │
        ▼
segment            pieces taken from the vocabulary
        │
        ▼
ids [12, 47, 9]    one integer per piece
        │
        ▼
embedding table    each id → a trained vector
```

The canonical string is the spelling that leaves normalization. It is the one form this tokenizer will split and look up. `"The  Cat  Sat"`, with capitals and a double space, and `"the cat sat"` become the same canonical string when the rules lowercase and collapse spaces. From pretokenization onward, the original capitals and the extra space are gone.

`"the cat sat"` is already in that form, so the same rules leave it unchanged. A string that contains a ligature, a fullwidth letter, or a combining accent is rewritten, which is why those inputs are in the diagram.

The rules are chosen when the tokenizer is built. They are not learned. Two tokenizers with different rules turn one raw string into different ids.

| Method | Base unit | Typical vocabulary | What an unknown word does |
|---|---|---|---|
| Word | whole word | tens or hundreds of thousands | becomes `<unk>` |
| Character | one character | about 100 to a few thousand | stays itself, if that character was kept |
| Byte-pair encoding | character or byte, then merges | about 30,000 to 100,000 | splits into known pieces |
| WordPiece | character, then merges | about 30,000 | splits, or `<unk>` if no character matches |
| Unigram / SentencePiece | substrings with probabilities | about 8,000 to 32,000 | falls back along the most probable pieces |
| Byte-level BPE | 256 bytes, then merges | about 30,000 to 200,000 | always splits into bytes |

---

## 2. Normalization

Normalization enters the raw string and leaves the canonical string. Unicode NFKC is the usual character-folding step. It maps compatibility characters onto ordinary ones.

```
ﬁ            →  fi
Ｔ            →  T
e + ´        →  é
```

Lowercasing and accent stripping are optional extra rules, applied before or after NFKC depending on the tokenizer.

```
"The  Cat  Sat"  → lowercase → "the  cat  sat"  → collapse spaces → "the cat sat"
"Café"           → NFKC → "Café"  → strip accents → "Cafe"  → lowercase → "cafe"
```

A GPT-style byte tokenizer often skips lowercasing and accent stripping. For that tokenizer, `"The"` and `"the"` are different canonical strings and can receive different ids. A BERT-style WordPiece tokenizer lowercases and strips accents, so those inputs share one canonical string and then one set of pieces.

Whatever string leaves this box is the only string the rest of the pipeline reads.

---

## 3. Pretokenization

Pretokenization enters the canonical string and leaves chunks. A merge learned later is not allowed to join characters across a chunk boundary.

* A word splitter leaves words and punctuation as separate chunks: `the`, `cat`, `sat`, `.`
* The GPT-2 splitter is a fixed regular expression. It peels off contractions, letter runs, number runs, and punctuation. A merge learned inside `cat` cannot absorb the space before it.
* SentencePiece does not cut the sentence into words first. It replaces each space with the character `▁` and keeps one stream: `▁the▁cat▁sat`.

```
canonical string "the cat sat"
        │
        ├── word split ──────────────►  the    cat    sat
        │
        └── SentencePiece space mark ►  ▁the ▁cat ▁sat
```

---

## 4. Word tokenization

Training counts words in the canonical corpus and keeps the most frequent ones, plus `<unk>`. Encoding looks each chunk up in a map from string to id.

```
vocabulary:  the→12   cat→47   sat→9   <unk>→0

"the cat sat"  →  [12, 47, 9]
"the kitten"   →  [12, 0]
```

`kitten` missed the frequency cutoff, so the lookup leaves the reserved id `0`. The id sequence is short. The vocabulary is large, and `kitten` shares no pieces with `cat`.

---

## 5. Character tokenization

The vocabulary is the set of characters kept from the canonical training text. Encoding walks that string one character at a time.

```
"cat"  →  c, a, t  →  [3, 1, 20]
```

A new word still produces ids, as long as its characters were kept. A word of five letters is five positions in the model.

---

## 6. Byte-pair encoding

Byte-pair encoding starts from characters of the canonical string and repeatedly glues the pair that occurs most often. The learned object is an ordered list of merges.

### Training

Each word below is written with an end-of-word marker `</w>`, so a piece that ends a word stays distinct from the same letters in the middle of a word.

```
low × 5     lowest × 2     newer × 6     wider × 3     new × 2
```

```
low     →  l o w </w>          × 5
lowest  →  l o w e s t </w>    × 2
newer   →  n e w e r </w>      × 6
wider   →  w i d e r </w>      × 3
new     →  n e w </w>          × 2
```

One step counts every adjacent pair, merges the winner, and rewrites the corpus. If two pairs have the same count, the pair that comes first in alphabetical order wins.

| Step | Pair merged | Count | Why this pair |
|---|---|---|---|
| 1 | `e` + `r` → `er` | 9 | tied with `r </w>`; `e r` sorts first |
| 2 | `er` + `</w>` → `er</w>` | 9 | only pair at this count |
| 3 | `e` + `w` → `ew` | 8 | tied with `n e`; `e w` sorts first |
| 4 | `n` + `ew` → `new` | 8 | only pair at this count |
| 5 | `l` + `o` → `lo` | 7 | tied with `o w`; `l o` sorts first |
| 6 | `lo` + `w` → `low` | 7 | only pair at this count |
| 7 | `new` + `er</w>` → `newer</w>` | 6 | only pair at this count |
| 8 | `low` + `</w>` → `low</w>` | 5 | only pair at this count |

After step 8 the corpus is:

```
low</w>                 × 5
low  e  s  t  </w>      × 2
newer</w>               × 6
w  i  d  er</w>         × 3
new  </w>               × 2
```

Training stops at a chosen number of merges. The vocabulary is the original characters, `</w>`, and each merged piece. The order of the merge list is part of the tokenizer.

```
counts ← empty map
for each word in the corpus, repeated by its frequency:
    symbols ← characters of the word, plus </w>
    add the word frequency to the count of every adjacent pair

repeat until the vocabulary is large enough:
    pair ← the pair with the highest count
            break ties by alphabetical order
    append pair to the merge list
    replace that adjacent pair with one new symbol
    recompute pair counts
```

### Encoding

Encoding applies the saved merges to one string. It does not recount the corpus. Among the pairs that sit in the string and also sit in the merge list, the earliest merge is glued. The search repeats.

```
low      →  low</w>
lowest   →  low   e   s   t   </w>
newer    →  newer</w>
lower    →  low   er</w>
widest   →  w  i  d  e  s  t  </w>
```

`lower` was not in the training count. The merges `low` and `er</w>` still apply, so the unseen word leaves two pieces instead of six characters. Those pieces are looked up and leave as integers.

```
function encode(word, merges):
    symbols ← characters of word, plus </w>
    repeat:
        candidates ← pairs inside symbols that appear in merges
        if candidates is empty: stop
        glue the candidate with the smallest merge index
    return symbols
```

Decoding concatenates the pieces and turns `</w>` back into a word boundary.

---

## 7. WordPiece

WordPiece also builds pieces by merging, but the pair it merges is the one that most increases the likelihood of the canonical training text. The score used in practice is:

```
score(a, b) = count(a, b) / (count(a) × count(b))
```

Encoding does not replay the merge list. From the start of the remaining string it takes the longest vocabulary piece that matches. BERT marks a continuation inside a word with `##`.

```
vocabulary pieces:  cat, ##s, c, a, t, s

"cats"  →  cat  +  ##s
"cat"   →  cat
```

If no piece matches at the current position, the word leaves as `<unk>`. The lookup is a trie: at each position, walk as far as the trie allows, emit that longest piece, and resume after it.

---

## 8. Unigram and SentencePiece

Unigram tokenization assigns a probability to every vocabulary piece. A split `p1 p2 p3` has probability `P(p1) × P(p2) × P(p3)`.

Training starts from a large seed vocabulary of frequent substrings of the canonical text. It fits the piece probabilities, then drops the pieces whose removal hurts the corpus likelihood the least. Characters stay, so a string can still be spelled out. The fit-and-drop cycle repeats until the vocabulary reaches the target size.

Encoding searches the possible splits with the Viterbi algorithm and leaves the split with the highest probability.

SentencePiece runs unigram segmentation, or byte-pair encoding, on the canonical string after each space has been rewritten as `▁`.

```
"the cat"  →  ▁the  ▁cat
```

Decoding replaces `▁` with a space. The same procedure trains on text that does not separate words with spaces.

---

## 9. Byte-level encoding

The base vocabulary is the 256 values of a byte. The canonical string is turned into UTF-8 bytes, and byte-pair encoding learns merges on those bytes.

```
"cat"  →  bytes 0x63 0x61 0x74  →  then whatever merges apply
```

A character such as `é` is more than one byte before any merge. A string absent from training still leaves ids, because every byte has a slot. GPT-2-style tokenizers and the Llama tokenizers are built this way. A pretokenizer regex may still run on the canonical string first, so merges do not cross the chunks that regex emits.

---

## 10. Special ids

A few ids are reserved when the vocabulary is built. They are not pieces discovered by counting the canonical text.

| Id name | Who inserts it | What training learns |
|---|---|---|
| `<bos>` | the program, at the first target position | the embedding row for that fixed id |
| `<eos>` | the program, at the end of a sequence | the embedding row |
| `<pad>` | the program, to fill a short row in a batch | usually ignored by the loss |
| `<unk>` | the encoder, when no piece matches | the embedding row |

If `<bos>` occupies slot `1`, every target sequence the program builds starts with `1`. Training learns the vector stored for id `1`. It is not asked to choose the integer.

---

## 11. What enters the embedding table

```
canonical string "the cat sat"
        │
        ▼
pieces the, cat, sat
        │
        ▼
ids [12, 47, 9]
        │
        ▼
rows E[12], E[47], E[9]     shape 3 × d_model
```

The embedding table has one row per vocabulary id. Row 12, row 47, and row 9 are stacked into the source matrix that enters the encoder. On the target side, `[1, 88]` for `<bos>` and `le` is read the same way. From this point the blocks see vectors, not characters.
