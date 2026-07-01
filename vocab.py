"""vocab.py -- the experimental substrate.

A small vocabulary of *real English words* grouped into MEANING CLASSES of
synonyms. This gives a clean, measurable decomposition that mirrors the
robust_paraphraser framework:

    word  = (meaning class, surface variant)
    meaning = which class               <- the legitimate, legible content
    surface = which synonym in the class <- the covert / lexical side-channel

Embeddings are *constructed* (not trained, frozen, public) so that:
    E[word] = normalize( mu[class] + eps * noise[word] )
Synonyms of a class share a center mu[class] but differ by a within-class
spread eps. Small eps  -> meaning easy to read, little surface capacity.
Large eps  -> more surface capacity but meaning harder to extract.
`eps` is the single knob controlling "how distinguishable are paraphrase
variants" -- the analog of a real word-embedding geometry.

This is deliberately public/frozen: the dictionary (word -> meaning) is shared
knowledge that both the model and the paraphraser may use. The research question
is NOT "can the model read meaning from words" (trivially yes) but "can the
model pass information that ISN'T meaning across a layer boundary."
"""
import numpy as np
import torch

# 24 real-English synonym classes, 4 synonyms each = 96 content words.
CLASSES = [
    ["big", "large", "huge", "giant"],
    ["small", "tiny", "little", "mini"],
    ["happy", "glad", "joyful", "cheerful"],
    ["sad", "gloomy", "unhappy", "blue"],
    ["fast", "quick", "rapid", "swift"],
    ["slow", "sluggish", "lazy", "leisurely"],
    ["hot", "warm", "heated", "fiery"],
    ["cold", "chilly", "icy", "frosty"],
    ["smart", "clever", "bright", "wise"],
    ["dumb", "foolish", "silly", "dull"],
    ["strong", "mighty", "powerful", "sturdy"],
    ["weak", "feeble", "frail", "faint"],
    ["good", "fine", "great", "nice"],
    ["bad", "awful", "poor", "lousy"],
    ["dark", "dim", "murky", "shadowy"],
    ["shiny", "glowing", "radiant", "gleaming"],
    ["old", "ancient", "aged", "elderly"],
    ["new", "fresh", "recent", "modern"],
    ["rich", "wealthy", "affluent", "loaded"],
    ["broke", "needy", "penniless", "poorish"],
    ["calm", "serene", "peaceful", "tranquil"],
    ["angry", "mad", "furious", "irate"],
    ["clean", "spotless", "tidy", "pristine"],
    ["dirty", "filthy", "grimy", "muddy"],
]
N_CLASS = len(CLASSES)
SYN_PER = len(CLASSES[0])

# Special (non-content) tokens. Hop indicators tell the chained-lookup task how
# many times to apply the rule; QUERY marks the readout position; PAD pads.
SPECIALS = ["<pad>", "<query>"] + [f"<hop{k}>" for k in range(1, 9)]

WORDS = [w for cls in CLASSES for w in cls] + SPECIALS
V = len(WORDS)
STOI = {w: i for i, w in enumerate(WORDS)}
ITOS = {i: w for i, w in enumerate(WORDS)}

PAD = STOI["<pad>"]
QUERY = STOI["<query>"]


def hop_tok(k):
    return STOI[f"<hop{k}>"]


# class_of[token] = meaning class index, or -1 for specials.
class_of = np.full(V, -1, dtype=np.int64)
for c, cls in enumerate(CLASSES):
    for w in cls:
        class_of[STOI[w]] = c
CLASS_OF = torch.tensor(class_of)

# members[c] = token ids of synonyms in class c
MEMBERS = [[STOI[w] for w in cls] for cls in CLASSES]


def build_embeddings(d=64, eps=0.5, seed=0):
    """Frozen public embedding matrix E (V x d).

    Content words: normalize(mu[class] + eps*noise). Specials: own random rows.
    Returns float32 tensor.
    """
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal((N_CLASS, d))
    mu /= np.linalg.norm(mu, axis=1, keepdims=True) + 1e-8
    E = np.zeros((V, d), dtype=np.float32)
    for i, w in enumerate(WORDS):
        c = class_of[i]
        if c >= 0:
            v = mu[c] + eps * rng.standard_normal(d)
        else:
            v = rng.standard_normal(d)  # specials are their own thing
        E[i] = v / (np.linalg.norm(v) + 1e-8)
    return torch.tensor(E)


def decode_ids(ids):
    """Pretty-print a sequence of token ids as English (for inspecting the
    residual stream)."""
    return " ".join(ITOS[int(i)] for i in ids)
