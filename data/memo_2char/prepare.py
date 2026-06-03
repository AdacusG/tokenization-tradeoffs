"""
Prepare the custom memorization dataset for Scenario B (2-character tokens).
Saves train.bin, val.bin containing the token IDs, and meta.pkl containing the
custom multi-character encoder and decoder mappings.
"""
import os
import pickle
import numpy as np

# Ensure paths align with your setup folder
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

# Fallback path if running outside the dataset directory
if not os.path.exists(input_file_path):
    input_file_path = 'memo_2char/input.txt'

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
print(f"Length of dataset in characters: {len(data):,}")

# --- 1. EXPLICITLY DEFINE THE SCENARIO B VOCABULARY ---
alphabet = ['a', 'b', 'c', 'd', 'e', 'f']

# Generate all 36 combinatoric pairs ('aa', 'ab', ..., 'ff')
combined_pairs = [f"{c1}{c2}" for c1 in alphabet for c2 in alphabet]

# Complete token vocabulary list (36 chunks + 2 structural symbols)
vocab = combined_pairs + ['=', '\n']
vocab_size = len(vocab)

print(f"All unique vocabulary tokens: {vocab}")
print(f"Vocab size (V'): {vocab_size}")

# --- 2. CREATE INTEGERS MAPPING LOOKUPS ---
stoi = { ch:i for i, ch in enumerate(vocab) }
itos = { i:ch for i, ch in enumerate(vocab) }

# --- 3. CUSTOM GREEDY PAIR-AWARE TOKENIZER ---
def encode(s):
    """Takes a string, slices it into 2-character tokens or symbols, outputs token IDs."""
    tokens = []
    i = 0
    n = len(s)
    
    while i < n:
        # Check if the current character is a standalone structural token
        if s[i] == '=' or s[i] == '\n':
            tokens.append(stoi[s[i]])
            i += 1
        # Check if the next two characters form one of our 36 pairs
        elif i + 1 < n and s[i] in alphabet and s[i+1] in alphabet:
            pair = s[i:i+2]
            tokens.append(stoi[pair])
            i += 2
        else:
            # Fallback block to safely skip spaces or unexpected characters
            i += 1
            
    return tokens

def decode(l):
    """Takes a list of integers, matches them to tokens, outputs a string."""
    return ''.join([itos[i] for i in l])


# --- 4. DATA SPLITS AND EXPORTION ---
# Split into train and validation sets (90% / 10%)
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# Process string data through the new multi-character chunking compiler
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"Train has {len(train_ids):,} compressed tokens")
print(f"Val has {len(val_ids):,} compressed tokens")

# Export token index arrays to binary files for model consumption
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# Save the metadata so sample.py and validation scripts can decode weights cleanly
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("🎉 Scenario B binaries and meta.pkl compiled successfully!")