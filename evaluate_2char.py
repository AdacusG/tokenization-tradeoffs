import os
import pickle
import torch
from model import GPTConfig, GPT

# 1. Set your desired vocabulary size (V) via command line argument or default to 26, keep it the same as your training script.
if len(sys.argv) > 1:
    try:
        V = int(sys.argv[1])
    except ValueError:
        print("Error: Vocabulary size V must be an integer (e.g., 6, 26, 52).")
        sys.exit(1)
else:
    V = 26  # Sensible default

# -----------------------------------------------------------------------------
checkpoint_dir = 'out-memo2'     # Your target checkpoint directory
data_dir = 'data/memo_2char'     # CHANGE THIS to your 2-character folder name!
input_path = 'input.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# -----------------------------------------------------------------------------

full_alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', "W", "X", "Y", "Z"]
base_chars = full_alphabet[:V]  # Use only the first V characters based on your vocab size

# 2. Build a lookup dictionary of true pairings from input.txt
true_pairs = {}
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Could not find {input_path}. Ensure it is in this folder.")

with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if '=' in line:
            prompt_side, target_side = line.split('=')
            true_pairs[prompt_side.strip()] = target_side.strip()

# 3. Generate all 36 2-character prompt combinations directly
test_prompts = [c1 + c2 for c1 in base_chars for c2 in base_chars] # ['aa', 'ab', ... 'ff']

# 4. Load your 2-Character Tokenizer mappings (meta.pkl)
meta_path = os.path.join(data_dir, 'meta.pkl')
if not os.path.exists(meta_path):
    raise FileNotFoundError(f"Could not find meta.pkl at {meta_path}. Check data_dir path!")

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']

# Helper lambda to turn tokens back into readable text strings
decode = lambda l: ''.join([itos[i] for i in l])

# 5. Load the trained 2-character model from your checkpoint folder
ckpt_path = os.path.join(checkpoint_dir, 'ckpt.pt')
print(f"Loading 2-Character checkpoint from {ckpt_path}...")
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']

# Clean up any potential DDP compiler prefixes
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()
model.to(device)

# 6. Evaluate all 36 pairings using direct dictionary lookups
correct_count = 0
total_count = len(test_prompts)

print(f"\nRunning 2-Character Accuracy Evaluation (Model Vocab Size: {gptconf.vocab_size})...")
print("-" * 60)

for prompt in test_prompts:
    try:
        # Step A: Direct lookup of the entire 2-char pair (e.g., 'aa')
        prompt_token_id = stoi[prompt]
        
        # Step B: Dynamically locate the equal sign key in your bigram dictionary
        # (Handles variations like '=', ' =', or similar artifacts)
        equals_key = [k for k in stoi.keys() if '=' in str(k)][0]
        equals_token_id = stoi[equals_key]
        
        # Combine them into your starting token list
        start_ids = [prompt_token_id, equals_token_id]
        
    except KeyError as e:
        print(f"Tokenizer Error: Could not find vocabulary keys for prompt '{prompt}'.")
        print(f"Make sure data_dir points to your 2-character data folder, not the 1-character one.")
        print(f"Sample of available keys in your loaded tokenizer: {list(stoi.keys())[:10]}")
        import sys; sys.exit(1)

    # Convert to tensor and send to GPU
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # A 2-character bigram model outputs the answer (e.g. 'ba') in exactly 1 token
    max_new_tokens = 1 
    
    with torch.no_grad():
        # High compression temp keeps selection completely greedy/deterministic
        y = model.generate(x, max_new_tokens, temperature=0.0001, top_k=5)
    
    # Decode integers back to text string
    full_output = decode(y[0].tolist())
    
    # Safely extract what the model generated right after the '=' symbol
    if '=' in full_output:
        predicted_target = full_output.split('=')[1].strip()
    else:
        # Fallback if text format squashed together
        predicted_target = full_output.replace(prompt, "").replace("=", "").strip()
        
    ground_truth = true_pairs[prompt]
    
    if predicted_target == ground_truth:
        correct_count += 1
        print(f"[CORRECT] {prompt}={predicted_target} (Expected: {ground_truth})")
    else:
        print(f"[FAIL] {prompt}={predicted_target} (Expected: {ground_truth})")

# 7. Print Final Summary Metrics
accuracy_pct = (correct_count / total_count) * 100
print("-" * 60)
print(f"2-CHAR MODEL TOTAL ACCURACY: {accuracy_pct:.2f}% ({correct_count}/{total_count} Correct)")
print("-" * 60)