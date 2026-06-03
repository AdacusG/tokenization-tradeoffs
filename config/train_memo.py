# Train a tiny "pico" GPT on our memorization task (Scenario A - Character-level)
# Run it from the root nanoGPT directory with: python train.py config/train_memo.py

vocab_size = 8
out_dir = 'out-memo'
eval_interval = 500   # Evaluate the model every 500 steps
eval_iters = 20
log_interval = 10     # Log training loss every 10 steps

# Always save checkpoints if the validation loss improves
always_save_checkpoint = True

dataset = 'memo_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 8        # Our sequences are short (e.g., "ab=db\n" is ~6 chars), 8 is plenty

# Crucial: Shrink the model layers down drastically
n_layer = 4
n_head = 4 
n_embd = 128
dropout = 0.0

learning_rate = 1e-3  # Fast learning rate for simple tasks
max_iters = 10000 # How many total training steps to run
lr_decay_iters = 10000 
min_lr = 1e-4
warmup_iters = 100
backend = 'gloo'      # Safe backend for Windows environments
compile = False