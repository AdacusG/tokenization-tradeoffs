# Train a tiny "pico" GPT on our memorization task
# Run it from the root nanoGPT directory with: python train.py config/train_memo.py
vocab_size = 8
out_dir = 'out-memo'
eval_interval = 250 # Evaluate the model every 250 steps
eval_iters = 20
log_interval = 10 # Log training loss every 10 steps

# Always save checkpoints if the validation loss improves
always_save_checkpoint = False

dataset = 'memo_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 64 # Our sequences are short (e.g., "ab=db\n" is ~6 chars), 8 is plenty

# Crucial: Shrink the model layers down drastically
n_layer = 6     # Only 2 transformer layers
n_head = 6      # 2 attention heads
n_embd = 384     # Hidden dimension of 32 (tiny!)
dropout = 0.0

learning_rate = 1e-3 # Fast learning rate for simple tasks
max_iters = 2000     # How many total training steps to run
lr_decay_iters = 2000
min_lr = 1e-4        # lr_decay_iters == max_iters means learning rate decays to min at the end
warmup_iters = 100
backend = 'gloo' # change to 'gloo' if on Windows, but often not needed for single CPU/GPU
compile = False