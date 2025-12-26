"""
Enhanced Diagnostic Chatbot GUI with Tkinter
=============================================

Extended visualizations showing:
- Chat window
- Attention heatmap (per layer)
- Token probabilities
- Generation trace
- NEW: Layer activations heatmap
- NEW: Entropy/confidence graph
- NEW: Embedding space visualization
- NEW: Residual stream changes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue
import math

# =============================================================================
# PART 1: Load Training Data
# =============================================================================

conversation_file = 'convo.txt'

try:
    with open(conversation_file, 'r') as f:
        training_conversations = f.read()
    print(f"Loaded training data from: {conversation_file}")
except FileNotFoundError:
    print(f"Error: Could not find {conversation_file}")
    print("Creating sample data...")
    
    training_conversations = """User: Hello
Bot: Hi there! How can I help you today?

User: What is your name?
Bot: I'm a chatbot assistant. Nice to meet you!

User: How are you?
Bot: I'm doing well, thank you for asking!

User: What is Python?
Bot: Python is a popular programming language known for its simplicity.

User: Thanks
Bot: You're welcome! Let me know if you need anything else.

User: Goodbye
Bot: Goodbye! Have a great day!

User: Tell me a joke
Bot: Why don't scientists trust atoms? Because they make up everything!

User: What can you do?
Bot: I can chat with you and answer simple questions.

User: Who created you?
Bot: I was created as a learning project to understand neural networks.

User: What is machine learning?
Bot: Machine learning is a type of AI that learns patterns from data.

User: How do neural networks work?
Bot: Neural networks process information through layers of connected nodes.

User: What is AI?
Bot: AI stands for Artificial Intelligence, technology that simulates human thinking.

User: Hello there
Bot: Hello! How are you doing today?

User: I'm good
Bot: That's great to hear!

User: Can you help me?
Bot: Of course! What do you need help with?

User: See you later
Bot: See you! Take care!

User: Good morning
Bot: Good morning! How can I assist you today?

User: What's your purpose?
Bot: My purpose is to demonstrate how neural networks learn to have conversations.

User: Are you real?
Bot: I'm a simple AI chatbot built for educational purposes.

User: What is deep learning?
Bot: Deep learning uses neural networks with many layers to learn complex patterns.

User: Explain transformers
Bot: Transformers are neural networks that use attention mechanisms to process sequences.

User: What is NLP?
Bot: NLP stands for Natural Language Processing, teaching computers to understand text.

User: How do you learn?
Bot: I learn by analyzing patterns in conversational data through training.

User: What is training?
Bot: Training is when a neural network adjusts its weights to minimize errors.

User: Tell me about science
Bot: Science is the systematic study of the natural world through observation and experimentation.

User: What's the weather?
Bot: I don't have access to weather data, but I hope it's nice!

User: Is it raining?
Bot: I can't check the weather, but you can look outside or use a weather app!

User: Thank you
Bot: You're very welcome! Happy to help.

User: Hi
Bot: Hi! What would you like to talk about?

User: Hey
Bot: Hey there! How can I help you?

User: What's up?
Bot: Not much! Just here to chat and answer questions.

User: Tell me another joke
Bot: Why do programmers prefer dark mode? Because light attracts bugs!

User: That's funny
Bot: Glad you enjoyed it!

User: What is coding?
Bot: Coding is writing instructions for computers to execute using programming languages.

User: Explain programming
Bot: Programming is creating software by writing code that tells computers what to do.

User: What is a computer?
Bot: A computer is an electronic device that processes data and executes programs.

User: How does a computer work?
Bot: Computers process information using processors, memory, and execute programmed instructions.

User: What is data?
Bot: Data is information that can be stored, processed, and analyzed by computers.
"""
    with open(conversation_file, 'w') as f:
        f.write(training_conversations)

# =============================================================================
# PART 2: Tokenization
# =============================================================================

def tokenize(text):
    text = re.sub(r'([.,!?:;\'])', r' \1 ', text)
    tokens = text.split()
    return tokens

all_tokens = tokenize(training_conversations)
vocab = sorted(list(set(all_tokens)))

SPECIAL_TOKENS = ['<PAD>', '<UNK>', '<START>', '<END>']
vocab = SPECIAL_TOKENS + vocab
vocab_size = len(vocab)

word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

def encode(text):
    tokens = tokenize(text)
    return [word_to_idx.get(word, word_to_idx['<UNK>']) for word in tokens]

def decode(indices):
    words = [idx_to_word.get(i, '<UNK>') for i in indices]
    words = [w for w in words if w not in SPECIAL_TOKENS]
    text = ' '.join(words)
    text = re.sub(r'\s+([.,!?:;\'])', r'\1', text)
    return text

data = torch.tensor(encode(training_conversations), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# =============================================================================
# PART 3: Model with Diagnostic Hooks
# =============================================================================

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.1)
        self.last_attn_weights = None
        # Store Q, K, V for visualization
        self.last_q = None
        self.last_k = None
        self.last_v = None
        self.last_scores = None  # Pre-softmax scores
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        # Store for visualization
        self.last_q = q.detach()
        self.last_k = k.detach()
        self.last_v = v.detach()
        
        scores = q @ k.transpose(-2, -1) * (C ** -0.5)
        self.last_scores = scores.detach()  # Store pre-softmax scores
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        self.last_attn_weights = attn.detach()
        attn = self.dropout(attn)
        
        out = attn @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
    def get_attention_weights(self):
        weights = [h.last_attn_weights for h in self.heads if h.last_attn_weights is not None]
        if weights:
            return torch.stack(weights).mean(dim=0)
        return None
    
    def get_individual_head_weights(self):
        """Get attention weights for each head separately (not averaged)"""
        weights = [h.last_attn_weights for h in self.heads]
        return weights  # List of [B, T, T] tensors, one per head
    
    def get_qkv_data(self):
        """Get Q, K, V tensors and pre-softmax scores for each head"""
        qkv_data = []
        for h in self.heads:
            qkv_data.append({
                'q': h.last_q,
                'k': h.last_k,
                'v': h.last_v,
                'scores': h.last_scores,
                'attn': h.last_attn_weights
            })
        return qkv_data

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(0.1)
        
        # Diagnostic: store activations
        self.last_hidden_activation = None
    
    def forward(self, x):
        hidden = self.linear1(x)
        activated = self.relu(hidden)
        self.last_hidden_activation = activated.detach()  # Store for visualization
        out = self.linear2(activated)
        out = self.dropout(out)
        return out

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.attn = MultiHeadAttention(n_embd, n_head, head_size, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
        # Diagnostic: store residual stream
        self.input_state = None
        self.post_attn_state = None
        self.output_state = None
    
    def forward(self, x):
        self.input_state = x.detach()
        
        attn_out = x + self.attn(self.ln1(x))
        self.post_attn_state = attn_out.detach()
        
        out = attn_out + self.ffwd(self.ln2(attn_out))
        self.output_state = out.detach()
        
        return out

class ChatbotModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # Diagnostics storage
        self.last_probs = None
        self.last_logits = None
        self.generation_trace = []
        self.entropy_trace = []
        self.layer_activations = []
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    def get_all_attention_weights(self):
        """Get attention from ALL layers"""
        return [block.attn.get_attention_weights() for block in self.blocks]
    
    def get_all_individual_head_weights(self):
        """Get individual head attention weights from ALL layers"""
        return [block.attn.get_individual_head_weights() for block in self.blocks]
    
    def get_all_qkv_data(self):
        """Get Q, K, V data from all layers"""
        return [block.attn.get_qkv_data() for block in self.blocks]
    
    def get_logit_lens(self, idx):
        """
        Logit Lens: Project intermediate representations to vocabulary space.
        Returns predictions at each layer as if that layer were the final layer.
        """
        B, T = idx.shape
        
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        logit_lens_results = []
        
        # After embedding (before any transformer blocks)
        embed_logits = self.lm_head(self.ln_f(x))
        embed_probs = F.softmax(embed_logits[:, -1, :], dim=-1)
        logit_lens_results.append({
            'stage': 'Embed',
            'logits': embed_logits[:, -1, :].detach(),
            'probs': embed_probs.detach()
        })
        
        # After each transformer block
        for i, block in enumerate(self.blocks):
            x = block(x)
            layer_logits = self.lm_head(self.ln_f(x))
            layer_probs = F.softmax(layer_logits[:, -1, :], dim=-1)
            logit_lens_results.append({
                'stage': f'Layer {i}',
                'logits': layer_logits[:, -1, :].detach(),
                'probs': layer_probs.detach()
            })
        
        return logit_lens_results
    
    def get_ffn_activations(self):
        """Get FFN activations from all layers"""
        return [block.ffwd.last_hidden_activation for block in self.blocks]
    
    def get_residual_stream(self):
        """Get residual stream states from all layers"""
        states = []
        for block in self.blocks:
            states.append({
                'input': block.input_state,
                'post_attn': block.post_attn_state,
                'output': block.output_state
            })
        return states
    
    def get_token_embeddings(self, indices):
        """Get embeddings for visualization"""
        return self.token_embedding(torch.tensor(indices)).detach()
    
    def generate_with_diagnostics(self, idx, max_new_tokens, temperature=1.0):
        """Generate with full diagnostic collection"""
        self.generation_trace = []
        self.entropy_trace = []
        self.layer_activations = []
        
        for step in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            
            self.last_probs = probs.detach()
            self.last_logits = logits.detach()
            
            # Calculate entropy (uncertainty measure)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()
            max_entropy = math.log(vocab_size)  # Maximum possible entropy
            normalized_entropy = entropy / max_entropy
            self.entropy_trace.append({
                'step': step,
                'entropy': entropy,
                'normalized': normalized_entropy,
                'confidence': 1 - normalized_entropy
            })
            
            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probs[0], min(5, probs.shape[1]))
            top_words = [(idx_to_word.get(i.item(), '?'), p.item()) for i, p in zip(top_indices, top_probs)]
            chosen_word = idx_to_word.get(idx_next.item(), '?')
            
            # Get logit lens data
            logit_lens = self.get_logit_lens(idx_cond)
            
            # Collect layer diagnostics
            layer_data = {
                'attention': self.get_all_attention_weights(),
                'individual_heads': self.get_all_individual_head_weights(),
                'qkv_data': self.get_all_qkv_data(),
                'ffn_activations': self.get_ffn_activations(),
                'residual_stream': self.get_residual_stream(),
                'logit_lens': logit_lens
            }
            self.layer_activations.append(layer_data)
            
            self.generation_trace.append({
                'step': step,
                'chosen': chosen_word,
                'chosen_idx': idx_next.item(),
                'top_candidates': top_words,
                'entropy': entropy,
                'confidence': 1 - normalized_entropy,
                'logits': logits[0].detach().clone(),
                'probs': probs[0].detach().clone()
            })
            
            # Stop at User or after sentence-ending punctuation
            if chosen_word == "User":
                break
            
            # Also stop if we just generated sentence-ending punctuation and have some content
            if chosen_word in ['.', '!', '?'] and step >= 3:
                break
            
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

# =============================================================================
# PART 4: Training
# =============================================================================

batch_size = 32  # Increased from 16 since we have lots of data
block_size = 32
n_embd = 128  # Increased from 48 for better capacity
n_head = 4
n_layer = 4   # Increased from 3 for deeper learning
learning_rate = 3e-4
max_iters = 5000  # Increased to 20k to fully utilize your large dataset
eval_interval = 500

print(f"\nVocabulary size: {vocab_size}")
print(f"Training tokens: {len(data)}")

model = ChatbotModel(vocab_size, n_embd, n_head, n_layer, block_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x, y

print("\nTraining...")
best_val_loss = float('inf')
best_model_state = None

for iter in range(max_iters):
    if iter % eval_interval == 0:
        model.eval()
        losses = []
        for _ in range(10):
            xb, yb = get_batch('val')
            with torch.no_grad():
                _, loss = model(xb, yb)
                losses.append(loss.item())
        val_loss = sum(losses) / len(losses)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"Step {iter}: val_loss = {val_loss:.4f} (best)")
        else:
            print(f"Step {iter}: val_loss = {val_loss:.4f}")
        
        model.train()
    
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
    optimizer.step()

if best_model_state:
    model.load_state_dict(best_model_state)
print(f"\nTraining complete! Best val_loss: {best_val_loss:.4f}")

# =============================================================================
# PART 5: Enhanced Tkinter GUI
# =============================================================================

class EnhancedDiagnosticGUI:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("Neural Network Chatbot - Enhanced Diagnostics")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a2e')
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#1a1a2e')
        style.configure('TLabel', background='#1a1a2e', foreground='#eee', font=('Consolas', 10))
        style.configure('Header.TLabel', font=('Consolas', 11, 'bold'), foreground='#00ff88')
        style.configure('TNotebook', background='#1a1a2e')
        style.configure('TNotebook.Tab', font=('Consolas', 9), padding=[10, 5])
        
        self.msg_queue = queue.Queue()
        self.current_layer = 0  # For layer selector
        
        self.setup_ui()
        self.process_queue()
    
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # =====================================================================
        # LEFT PANEL - Chat
        # =====================================================================
        left_frame = ttk.Frame(main_frame, width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        left_frame.pack_propagate(False)
        
        ttk.Label(left_frame, text="💬 CHAT", style='Header.TLabel').pack(anchor='w')
        
        self.chat_display = scrolledtext.ScrolledText(
            left_frame, wrap=tk.WORD, width=45, height=35,
            font=('Consolas', 11), bg='#16213e', fg='white',
            insertbackground='white', selectbackground='#404040'
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=5)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.tag_configure('user', foreground='#69b4ff')
        self.chat_display.tag_configure('bot', foreground='#00ff88')
        self.chat_display.tag_configure('system', foreground='#888888')
        
        # Input area
        input_frame = ttk.Frame(left_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        self.input_entry = tk.Entry(
            input_frame, font=('Consolas', 11),
            bg='#0f3460', fg='white', insertbackground='white'
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.input_entry.bind('<Return>', self.send_message)
        
        send_btn = tk.Button(
            input_frame, text="Send", command=self.send_message,
            font=('Consolas', 10), bg='#00aa66', fg='white'
        )
        send_btn.pack(side=tk.RIGHT)
        
        # =====================================================================
        # RIGHT PANEL - Diagnostics (Tabbed)
        # =====================================================================
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # ----- TAB 1: Attention & Predictions -----
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="📊 Attention & Predictions")
        
        # Attention section
        attn_frame = ttk.Frame(tab1)
        attn_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        attn_header = ttk.Frame(attn_frame)
        attn_header.pack(fill=tk.X)
        ttk.Label(attn_header, text="🔍 ATTENTION WEIGHTS", style='Header.TLabel').pack(side=tk.LEFT)
        
        # Layer selector
        ttk.Label(attn_header, text="  Layer: ").pack(side=tk.LEFT)
        self.layer_var = tk.StringVar(value="0")
        self.layer_selector = ttk.Combobox(
            attn_header, textvariable=self.layer_var,
            values=[str(i) for i in range(n_layer)], width=5, state='readonly'
        )
        self.layer_selector.pack(side=tk.LEFT)
        self.layer_selector.bind('<<ComboboxSelected>>', self.on_layer_change)
        
        self.attn_canvas = tk.Canvas(attn_frame, bg='#16213e', height=180, highlightthickness=0)
        self.attn_canvas.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Predictions section
        prob_frame = ttk.Frame(tab1)
        prob_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Label(prob_frame, text="📈 TOP PREDICTIONS", style='Header.TLabel').pack(anchor='w')
        
        self.prob_canvas = tk.Canvas(prob_frame, bg='#16213e', height=150, highlightthickness=0)
        self.prob_canvas.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # ----- TAB 2: Generation Trace -----
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="📝 Generation Trace")
        
        ttk.Label(tab2, text="📝 STEP-BY-STEP GENERATION", style='Header.TLabel').pack(anchor='w', pady=5)
        
        self.trace_display = scrolledtext.ScrolledText(
            tab2, wrap=tk.WORD, width=70, height=30,
            font=('Consolas', 10), bg='#16213e', fg='#cccccc'
        )
        self.trace_display.pack(fill=tk.BOTH, expand=True, pady=5)
        self.trace_display.config(state=tk.DISABLED)
        self.trace_display.tag_configure('chosen', foreground='#00ff88', font=('Consolas', 10, 'bold'))
        self.trace_display.tag_configure('step', foreground='#ffaa00')
        self.trace_display.tag_configure('prob', foreground='#69b4ff')
        self.trace_display.tag_configure('entropy', foreground='#ff6b6b')
        
        # ----- TAB 3: Confidence/Entropy -----
        tab3 = ttk.Frame(self.notebook)
        self.notebook.add(tab3, text="📉 Confidence")
        
        ttk.Label(tab3, text="📉 CONFIDENCE OVER GENERATION", style='Header.TLabel').pack(anchor='w', pady=5)
        ttk.Label(tab3, text="(Higher = more certain about word choice)", 
                  foreground='#888888', background='#1a1a2e').pack(anchor='w')
        
        self.confidence_canvas = tk.Canvas(tab3, bg='#16213e', height=250, highlightthickness=0)
        self.confidence_canvas.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # ----- TAB 4: FFN Activations -----
        tab4 = ttk.Frame(self.notebook)
        self.notebook.add(tab4, text="🧠 Neuron Activations")
        
        ttk.Label(tab4, text="🧠 FEED-FORWARD NETWORK ACTIVATIONS", style='Header.TLabel').pack(anchor='w', pady=5)
        ttk.Label(tab4, text="(Which neurons fire for the last token)", 
                  foreground='#888888', background='#1a1a2e').pack(anchor='w')
        
        # Layer selector for FFN
        ffn_header = ttk.Frame(tab4)
        ffn_header.pack(fill=tk.X, pady=5)
        ttk.Label(ffn_header, text="Layer: ").pack(side=tk.LEFT)
        self.ffn_layer_var = tk.StringVar(value="0")
        self.ffn_layer_selector = ttk.Combobox(
            ffn_header, textvariable=self.ffn_layer_var,
            values=[str(i) for i in range(n_layer)], width=5, state='readonly'
        )
        self.ffn_layer_selector.pack(side=tk.LEFT)
        self.ffn_layer_selector.bind('<<ComboboxSelected>>', self.on_ffn_layer_change)
        
        self.ffn_canvas = tk.Canvas(tab4, bg='#16213e', height=300, highlightthickness=0)
        self.ffn_canvas.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # ----- TAB 5: Residual Stream -----
        tab5 = ttk.Frame(self.notebook)
        self.notebook.add(tab5, text="🌊 Residual Stream")
        
        ttk.Label(tab5, text="🌊 RESIDUAL STREAM MAGNITUDE", style='Header.TLabel').pack(anchor='w', pady=5)
        ttk.Label(tab5, text="(How representation changes through layers)", 
                  foreground='#888888', background='#1a1a2e').pack(anchor='w')
        
        self.residual_canvas = tk.Canvas(tab5, bg='#16213e', height=300, highlightthickness=0)
        self.residual_canvas.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # ----- TAB 6: Logits View -----
        tab6 = ttk.Frame(self.notebook)
        self.notebook.add(tab6, text="📊 Raw Logits")
        
        ttk.Label(tab6, text="📊 RAW LOGITS (Pre-Softmax Scores)", style='Header.TLabel').pack(anchor='w', pady=5)
        ttk.Label(tab6, text="(Model's raw preferences before probability conversion)", 
                  foreground='#888888', background='#1a1a2e').pack(anchor='w')
        
        self.logits_canvas = tk.Canvas(tab6, bg='#16213e', height=300, highlightthickness=0)
        self.logits_canvas.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # ----- TAB 7: Logit Lens -----
        tab7 = ttk.Frame(self.notebook)
        self.notebook.add(tab7, text="🔬 Logit Lens")
        
        ttk.Label(tab7, text="🔬 LOGIT LENS", style='Header.TLabel').pack(anchor='w', pady=5)
        ttk.Label(tab7, text="(What the model would predict if stopped at each layer)", 
                  foreground='#888888', background='#1a1a2e').pack(anchor='w')
        
        self.logit_lens_canvas = tk.Canvas(tab7, bg='#16213e', height=350, highlightthickness=0)
        self.logit_lens_canvas.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # ----- TAB 8: Q, K, V Inspector -----
        tab8 = ttk.Frame(self.notebook)
        self.notebook.add(tab8, text="🔑 Q, K, V Inspector")
        
        ttk.Label(tab8, text="🔑 QUERY, KEY, VALUE INSPECTOR", style='Header.TLabel').pack(anchor='w', pady=5)
        
        # Layer and Head selectors
        qkv_header = ttk.Frame(tab8)
        qkv_header.pack(fill=tk.X, pady=5)
        ttk.Label(qkv_header, text="Layer: ").pack(side=tk.LEFT)
        self.qkv_layer_var = tk.StringVar(value="0")
        self.qkv_layer_selector = ttk.Combobox(
            qkv_header, textvariable=self.qkv_layer_var,
            values=[str(i) for i in range(n_layer)], width=5, state='readonly'
        )
        self.qkv_layer_selector.pack(side=tk.LEFT, padx=(0, 15))
        self.qkv_layer_selector.bind('<<ComboboxSelected>>', self.on_qkv_change)
        
        ttk.Label(qkv_header, text="Head: ").pack(side=tk.LEFT)
        self.qkv_head_var = tk.StringVar(value="0")
        self.qkv_head_selector = ttk.Combobox(
            qkv_header, textvariable=self.qkv_head_var,
            values=[str(i) for i in range(n_head)], width=5, state='readonly'
        )
        self.qkv_head_selector.pack(side=tk.LEFT)
        self.qkv_head_selector.bind('<<ComboboxSelected>>', self.on_qkv_change)
        
        # Scrollable canvas for QKV visualization
        qkv_container = ttk.Frame(tab8)
        qkv_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.qkv_canvas = tk.Canvas(qkv_container, bg='#16213e', height=500, highlightthickness=0)
        qkv_scrollbar = ttk.Scrollbar(qkv_container, orient=tk.VERTICAL, command=self.qkv_canvas.yview)
        self.qkv_canvas.configure(yscrollcommand=qkv_scrollbar.set)
        
        qkv_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.qkv_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Welcome message
        self.add_chat_message("System", "Enhanced Diagnostic Chatbot ready!", 'system')
        self.add_chat_message("System", f"Model: {total_params:,} params, {n_layer} layers, vocab: {vocab_size}", 'system')
        
        # Store diagnostic data
        self.last_trace = []
        self.last_layer_data = []
        self.last_context = []
        
        # Tokens to filter from displays (placeholders that clutter visualization)
        self.filter_tokens = {'User', 'User:', 'Bot', 'Bot:', ':', '?'}
    
    def filter_display_tokens(self, token_indices):
        """
        Filter out placeholder tokens from display.
        Returns: (filtered_tokens, filtered_indices, original_positions)
        - filtered_tokens: list of token strings to display
        - filtered_indices: indices into the original sequence (for matrix slicing)
        """
        tokens = [idx_to_word.get(i, '?') for i in token_indices]
        filtered = [(i, tok) for i, tok in enumerate(tokens) if tok not in self.filter_tokens]
        
        if not filtered:
            return tokens, list(range(len(tokens)))  # Return all if filtering removes everything
        
        filtered_indices, filtered_tokens = zip(*filtered)
        return list(filtered_tokens), list(filtered_indices)
    
    def on_layer_change(self, event=None):
        """Redraw attention for selected layer"""
        if self.last_trace and self.last_layer_data:
            self.update_attention_for_layer(int(self.layer_var.get()))
    
    def on_ffn_layer_change(self, event=None):
        """Redraw FFN activations for selected layer"""
        if self.last_layer_data:
            self.update_ffn_for_layer(int(self.ffn_layer_var.get()))
    
    def on_qkv_change(self, event=None):
        """Redraw Q, K, V visualization for selected layer and head"""
        if self.last_layer_data:
            self.update_qkv_inspector(int(self.qkv_layer_var.get()), int(self.qkv_head_var.get()))
    
    def add_chat_message(self, sender, message, tag):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{sender}: ", tag)
        self.chat_display.insert(tk.END, f"{message}\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def send_message(self, event=None):
        user_input = self.input_entry.get().strip()
        if not user_input:
            return
        
        self.input_entry.delete(0, tk.END)
        self.add_chat_message("You", user_input, 'user')
        
        thread = threading.Thread(target=self.generate_response, args=(user_input,))
        thread.daemon = True
        thread.start()
    
    def generate_response(self, user_input):
        try:
            self.model.eval()
            
            # Build prompt with more context from training data
            prompt = f"User: {user_input}\nBot:"
            context = encode(prompt)
            context_tensor = torch.tensor([context], dtype=torch.long)
            
            with torch.no_grad():
                generated = self.model.generate_with_diagnostics(
                    context_tensor, max_new_tokens=20, temperature=0.6
                )
            
            full_response = decode(generated[0].tolist())
            
            bot_response = "(no response)"
            if "Bot:" in full_response:
                parts = full_response.split("Bot:")
                if len(parts) > 1:
                    bot_response = parts[-1].strip()
                    if "User:" in bot_response:
                        bot_response = bot_response[:bot_response.find("User:")].strip()
            
            # Queue all UI updates
            self.msg_queue.put(('chat', bot_response))
            self.msg_queue.put(('trace', self.model.generation_trace))
            self.msg_queue.put(('store_data', self.model.generation_trace, 
                               self.model.layer_activations, context))
            self.msg_queue.put(('attention', 0))  # Default to layer 0
            self.msg_queue.put(('probs', self.model.generation_trace))
            self.msg_queue.put(('confidence', self.model.entropy_trace))
            self.msg_queue.put(('ffn', 0))  # Default to layer 0
            self.msg_queue.put(('residual', self.model.layer_activations))
            self.msg_queue.put(('logits', self.model.generation_trace))
            self.msg_queue.put(('logit_lens', self.model.layer_activations))
            self.msg_queue.put(('qkv_inspector', 0, 0))  # Default to layer 0, head 0
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.msg_queue.put(('chat', f"Error: {str(e)}"))
    
    def process_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                msg_type = msg[0]
                
                if msg_type == 'chat':
                    self.add_chat_message("Bot", msg[1], 'bot')
                elif msg_type == 'trace':
                    self.update_trace(msg[1])
                elif msg_type == 'store_data':
                    self.last_trace = msg[1]
                    self.last_layer_data = msg[2]
                    self.last_context = msg[3]
                elif msg_type == 'attention':
                    self.update_attention_for_layer(msg[1])
                elif msg_type == 'probs':
                    self.update_probabilities(msg[1])
                elif msg_type == 'confidence':
                    self.update_confidence_graph(msg[1])
                elif msg_type == 'ffn':
                    self.update_ffn_for_layer(msg[1])
                elif msg_type == 'residual':
                    self.update_residual_stream(msg[1])
                elif msg_type == 'logits':
                    self.update_logits_view(msg[1])
                elif msg_type == 'logit_lens':
                    self.update_logit_lens(msg[1])
                elif msg_type == 'qkv_inspector':
                    self.update_qkv_inspector(msg[1], msg[2])
                    
        except queue.Empty:
            pass
        
        self.root.after(50, self.process_queue)
    
    def update_trace(self, trace):
        self.trace_display.config(state=tk.NORMAL)
        self.trace_display.delete(1.0, tk.END)
        
        for step_info in trace:
            step = step_info['step']
            chosen = step_info['chosen']
            candidates = step_info['top_candidates']
            confidence = step_info['confidence']
            entropy = step_info['entropy']
            
            # Skip displaying steps where chosen token is a placeholder
            if chosen in self.filter_tokens:
                continue
            
            self.trace_display.insert(tk.END, f"Step {step}: ", 'step')
            self.trace_display.insert(tk.END, f"'{chosen}' ", 'chosen')
            
            # Confidence indicator
            conf_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            self.trace_display.insert(tk.END, f"[{conf_bar}] ", 'entropy')
            self.trace_display.insert(tk.END, f"{confidence:.1%}\n", 'entropy')
            
            # Alternatives - filter out placeholder tokens
            alts = [f"{w}({p:.2f})" for w, p in candidates[:4] if w != chosen and w not in self.filter_tokens]
            if alts:
                self.trace_display.insert(tk.END, f"        Alternatives: {', '.join(alts)}\n", 'prob')
            
            self.trace_display.insert(tk.END, "\n")
        
        self.trace_display.see(tk.END)
        self.trace_display.config(state=tk.DISABLED)
    
    def update_attention_for_layer(self, layer_idx):
        self.attn_canvas.delete("all")
        
        if not self.last_layer_data or not self.last_trace:
            self.attn_canvas.create_text(200, 90, text="No data yet", fill='#666')
            return
        
        # Get attention from specified layer for last generation step
        last_step_data = self.last_layer_data[-1] if self.last_layer_data else None
        if not last_step_data:
            return
        
        attn_weights = last_step_data['attention'][layer_idx]
        if attn_weights is None:
            return
        
        attn = attn_weights[0]  # First batch
        
        # Filter out placeholder tokens
        context_indices = self.last_context[-block_size:]
        all_tokens = [idx_to_word.get(i, '?') for i in context_indices]
        
        # Get filtered tokens and their original indices
        filtered_tokens, filtered_indices = self.filter_display_tokens(context_indices)
        n_tokens = min(len(filtered_tokens), 14)
        
        if n_tokens == 0:
            return
        
        canvas_width = self.attn_canvas.winfo_width() or 500
        cell_width = min(35, (canvas_width - 80) // n_tokens)
        start_x = 60
        start_y = 30
        
        self.attn_canvas.create_text(
            start_x, 10, text=f"Layer {layer_idx} attention for last token:",
            fill='#aaa', font=('Consolas', 9), anchor='w'
        )
        
        # Get attention weights for filtered token positions
        last_attn_full = attn[-1, :].numpy()
        
        for i in range(n_tokens):
            orig_idx = filtered_indices[-(n_tokens - i)]  # Get from end of filtered list
            token = filtered_tokens[-(n_tokens - i)]
            weight = last_attn_full[orig_idx] if orig_idx < len(last_attn_full) else 0
            
            x = start_x + i * cell_width
            y = start_y
            
            intensity = int(min(weight * 3, 1.0) * 255)  # Scale up for visibility
            color = f'#{intensity:02x}{min(255, intensity + 30):02x}{intensity:02x}'
            
            self.attn_canvas.create_rectangle(
                x, y, x + cell_width - 2, y + 25,
                fill=color, outline='#333'
            )
            
            display_token = token[:5] if len(token) > 5 else token
            self.attn_canvas.create_text(
                x + cell_width // 2, y + 35,
                text=display_token, fill='#aaa', font=('Consolas', 7)
            )
            
            self.attn_canvas.create_text(
                x + cell_width // 2, y + 12,
                text=f"{weight:.2f}",
                fill='black' if intensity > 128 else 'white',
                font=('Consolas', 7)
            )
    
    def update_probabilities(self, trace):
        self.prob_canvas.delete("all")
        
        if not trace:
            return
        
        last_step = trace[-1]
        # Filter out placeholder tokens from candidates
        candidates = [(w, p) for w, p in last_step['top_candidates'][:10] if w not in self.filter_tokens][:6]
        chosen = last_step['chosen']
        
        if not candidates:
            return
        
        canvas_width = self.prob_canvas.winfo_width() or 400
        bar_height = 18
        max_bar_width = canvas_width - 140
        start_x = 90
        start_y = 15
        
        max_prob = max(p for _, p in candidates) if candidates else 1.0
        
        for i, (word, prob) in enumerate(candidates):
            y = start_y + i * (bar_height + 5)
            bar_width = (prob / max_prob) * max_bar_width
            
            display_word = word[:8] if len(word) > 8 else word
            self.prob_canvas.create_text(
                start_x - 5, y + bar_height // 2,
                text=display_word, fill='white', font=('Consolas', 9), anchor='e'
            )
            
            color = '#00ff88' if word == chosen else '#4488ff'
            self.prob_canvas.create_rectangle(
                start_x, y, start_x + bar_width, y + bar_height,
                fill=color, outline='#222'
            )
            
            self.prob_canvas.create_text(
                start_x + bar_width + 5, y + bar_height // 2,
                text=f"{prob:.3f}", fill='#aaa', font=('Consolas', 9), anchor='w'
            )
    
    def update_confidence_graph(self, entropy_trace):
        self.confidence_canvas.delete("all")
        
        if not entropy_trace:
            self.confidence_canvas.create_text(200, 125, text="No data yet", fill='#666')
            return
        
        canvas_width = self.confidence_canvas.winfo_width() or 600
        canvas_height = self.confidence_canvas.winfo_height() or 250
        
        padding = 50
        graph_width = canvas_width - 2 * padding
        graph_height = canvas_height - 2 * padding
        
        # Draw axes
        self.confidence_canvas.create_line(
            padding, padding, padding, canvas_height - padding,
            fill='#444', width=2
        )
        self.confidence_canvas.create_line(
            padding, canvas_height - padding, 
            canvas_width - padding, canvas_height - padding,
            fill='#444', width=2
        )
        
        # Labels
        self.confidence_canvas.create_text(
            padding - 10, padding, text="100%", fill='#888', font=('Consolas', 8), anchor='e'
        )
        self.confidence_canvas.create_text(
            padding - 10, canvas_height - padding, text="0%", fill='#888', font=('Consolas', 8), anchor='e'
        )
        self.confidence_canvas.create_text(
            canvas_width // 2, canvas_height - 15, text="Generation Step", 
            fill='#888', font=('Consolas', 9)
        )
        
        # Plot confidence line
        n_points = len(entropy_trace)
        if n_points < 2:
            return
        
        points = []
        for i, data in enumerate(entropy_trace):
            x = padding + (i / (n_points - 1)) * graph_width
            y = canvas_height - padding - data['confidence'] * graph_height
            points.append((x, y))
        
        # Draw line
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            self.confidence_canvas.create_line(
                x1, y1, x2, y2, fill='#00ff88', width=2
            )
        
        # Draw points
        for i, (x, y) in enumerate(points):
            color = '#00ff88' if entropy_trace[i]['confidence'] > 0.5 else '#ff6b6b'
            self.confidence_canvas.create_oval(
                x - 4, y - 4, x + 4, y + 4,
                fill=color, outline='white'
            )
            
            # Step number
            self.confidence_canvas.create_text(
                x, canvas_height - padding + 15,
                text=str(i), fill='#666', font=('Consolas', 8)
            )
    
    def update_ffn_for_layer(self, layer_idx):
        self.ffn_canvas.delete("all")
        
        if not self.last_layer_data:
            self.ffn_canvas.create_text(200, 150, text="No data yet", fill='#666')
            return
        
        last_step_data = self.last_layer_data[-1] if self.last_layer_data else None
        if not last_step_data:
            return
        
        ffn_act = last_step_data['ffn_activations'][layer_idx]
        if ffn_act is None:
            return
        
        # Get activations for last token
        activations = ffn_act[0, -1, :].numpy()  # [hidden_dim]
        
        canvas_width = self.ffn_canvas.winfo_width() or 600
        canvas_height = self.ffn_canvas.winfo_height() or 300
        
        # Show as heatmap grid
        n_neurons = len(activations)
        cols = int(math.sqrt(n_neurons) * 1.5)
        rows = math.ceil(n_neurons / cols)
        
        cell_size = min((canvas_width - 40) // cols, (canvas_height - 60) // rows, 15)
        start_x = 20
        start_y = 30
        
        self.ffn_canvas.create_text(
            start_x, 10, 
            text=f"Layer {layer_idx} FFN activations ({n_neurons} neurons, ReLU output)",
            fill='#aaa', font=('Consolas', 9), anchor='w'
        )
        
        max_act = max(abs(activations.max()), abs(activations.min()), 1e-6)
        
        for i, act in enumerate(activations):
            row = i // cols
            col = i % cols
            x = start_x + col * cell_size
            y = start_y + row * cell_size
            
            # Normalize and color
            norm_act = act / max_act
            if norm_act > 0:
                intensity = int(norm_act * 255)
                color = f'#{intensity:02x}{intensity:02x}00'  # Yellow for positive
            else:
                color = '#111'  # Dark for zero/negative (ReLU)
            
            self.ffn_canvas.create_rectangle(
                x, y, x + cell_size - 1, y + cell_size - 1,
                fill=color, outline=''
            )
        
        # Legend
        legend_y = start_y + rows * cell_size + 15
        self.ffn_canvas.create_text(
            start_x, legend_y, text="Dark = inactive (0)", fill='#666', 
            font=('Consolas', 8), anchor='w'
        )
        self.ffn_canvas.create_text(
            start_x + 150, legend_y, text="Yellow = active (high)", fill='#ffff00', 
            font=('Consolas', 8), anchor='w'
        )
    
    def update_residual_stream(self, layer_data):
        self.residual_canvas.delete("all")
        
        if not layer_data:
            self.residual_canvas.create_text(200, 150, text="No data yet", fill='#666')
            return
        
        canvas_width = self.residual_canvas.winfo_width() or 600
        canvas_height = self.residual_canvas.winfo_height() or 300
        
        padding = 60
        
        # Get last step's residual data
        last_data = layer_data[-1]
        residual = last_data['residual_stream']
        
        # Calculate magnitude at each stage
        stages = []
        labels = []
        
        for i, layer_res in enumerate(residual):
            # Magnitude of last token's representation
            input_mag = layer_res['input'][0, -1, :].norm().item()
            post_attn_mag = layer_res['post_attn'][0, -1, :].norm().item()
            output_mag = layer_res['output'][0, -1, :].norm().item()
            
            if i == 0:
                stages.append(input_mag)
                labels.append("Embed")
            stages.append(post_attn_mag)
            labels.append(f"L{i}+Attn")
            stages.append(output_mag)
            labels.append(f"L{i}+FFN")
        
        if not stages:
            return
        
        # Draw bar chart
        bar_width = (canvas_width - 2 * padding) / len(stages) - 5
        max_val = max(stages)
        
        self.residual_canvas.create_text(
            padding, 15, text="Representation magnitude through network:",
            fill='#aaa', font=('Consolas', 9), anchor='w'
        )
        
        for i, (val, label) in enumerate(zip(stages, labels)):
            x = padding + i * (bar_width + 5)
            bar_height = (val / max_val) * (canvas_height - 2 * padding - 30)
            y = canvas_height - padding - bar_height
            
            # Color gradient based on layer
            hue = int(120 + (i / len(stages)) * 120)  # Green to cyan
            color = f'#{0:02x}{hue:02x}{hue:02x}'
            
            self.residual_canvas.create_rectangle(
                x, y, x + bar_width, canvas_height - padding,
                fill=color, outline='#333'
            )
            
            # Value
            self.residual_canvas.create_text(
                x + bar_width / 2, y - 10,
                text=f"{val:.1f}", fill='#aaa', font=('Consolas', 7)
            )
            
            # Label
            self.residual_canvas.create_text(
                x + bar_width / 2, canvas_height - padding + 12,
                text=label, fill='#888', font=('Consolas', 7)
            )
    
    def update_logits_view(self, trace):
        self.logits_canvas.delete("all")
        
        if not trace:
            self.logits_canvas.create_text(200, 150, text="No data yet", fill='#666')
            return
        
        last_step = trace[-1]
        logits = last_step['logits'].numpy()
        
        canvas_width = self.logits_canvas.winfo_width() or 600
        canvas_height = self.logits_canvas.winfo_height() or 300
        
        # Show top and bottom logits
        top_k = 8
        
        # Get indices sorted by logits, filtering out placeholder tokens
        sorted_indices = logits.argsort()[::-1]  # Descending
        top_indices = []
        for idx in sorted_indices:
            word = idx_to_word.get(idx, '?')
            if word not in self.filter_tokens:
                top_indices.append(idx)
                if len(top_indices) >= top_k:
                    break
        
        sorted_indices_asc = logits.argsort()  # Ascending
        bottom_indices = []
        for idx in sorted_indices_asc:
            word = idx_to_word.get(idx, '?')
            if word not in self.filter_tokens:
                bottom_indices.append(idx)
                if len(bottom_indices) >= top_k:
                    break
        
        padding = 50
        bar_height = 18
        max_abs = max(abs(logits.max()), abs(logits.min()))
        
        self.logits_canvas.create_text(
            padding, 15, text="Raw logits (pre-softmax) - Top positive and negative:",
            fill='#aaa', font=('Consolas', 9), anchor='w'
        )
        
        # Draw top logits (positive)
        self.logits_canvas.create_text(
            padding, 40, text="Highest (preferred):", fill='#00ff88', 
            font=('Consolas', 8), anchor='w'
        )
        
        center_x = canvas_width // 2
        max_bar = (canvas_width - 2 * padding) // 2 - 50
        
        for i, idx in enumerate(top_indices):
            y = 55 + i * (bar_height + 3)
            word = idx_to_word.get(idx, '?')
            val = logits[idx]
            
            bar_width = (val / max_abs) * max_bar
            
            self.logits_canvas.create_text(
                center_x - 5, y + bar_height // 2,
                text=word[:8], fill='white', font=('Consolas', 8), anchor='e'
            )
            
            self.logits_canvas.create_rectangle(
                center_x, y, center_x + bar_width, y + bar_height,
                fill='#00aa66', outline='#333'
            )
            
            self.logits_canvas.create_text(
                center_x + bar_width + 5, y + bar_height // 2,
                text=f"{val:.1f}", fill='#888', font=('Consolas', 8), anchor='w'
            )
        
        # Draw bottom logits (negative)
        start_y = 55 + top_k * (bar_height + 3) + 20
        self.logits_canvas.create_text(
            padding, start_y, text="Lowest (avoided):", fill='#ff6b6b', 
            font=('Consolas', 8), anchor='w'
        )
        
        for i, idx in enumerate(bottom_indices):
            y = start_y + 15 + i * (bar_height + 3)
            word = idx_to_word.get(idx, '?')
            val = logits[idx]
            
            bar_width = abs(val / max_abs) * max_bar
            
            self.logits_canvas.create_text(
                center_x - 5, y + bar_height // 2,
                text=word[:8], fill='white', font=('Consolas', 8), anchor='e'
            )
            
            self.logits_canvas.create_rectangle(
                center_x - bar_width, y, center_x, y + bar_height,
                fill='#aa3333', outline='#333'
            )
            
            self.logits_canvas.create_text(
                center_x - bar_width - 5, y + bar_height // 2,
                text=f"{val:.1f}", fill='#888', font=('Consolas', 8), anchor='e'
            )
    
    def update_logit_lens(self, layer_data):
        """Visualize what the model would predict at each layer"""
        self.logit_lens_canvas.delete("all")
        
        if not layer_data:
            self.logit_lens_canvas.create_text(200, 150, text="No data yet", fill='#666')
            return
        
        # Get logit lens data from the last generation step
        last_step_data = layer_data[-1] if layer_data else None
        if not last_step_data or 'logit_lens' not in last_step_data:
            self.logit_lens_canvas.create_text(200, 150, text="No logit lens data", fill='#666')
            return
        
        logit_lens = last_step_data['logit_lens']
        
        canvas_width = self.logit_lens_canvas.winfo_width() or 700
        canvas_height = self.logit_lens_canvas.winfo_height() or 350
        
        n_stages = len(logit_lens)
        stage_width = (canvas_width - 80) // n_stages
        top_k = 5  # Show top 5 predictions per stage
        
        padding_top = 40
        bar_height = 22
        
        self.logit_lens_canvas.create_text(
            40, 15, text="Top predictions at each stage (watch the answer form!):",
            fill='#aaa', font=('Consolas', 9), anchor='w'
        )
        
        # Track the final top prediction (excluding filtered tokens) to highlight its journey
        final_probs = logit_lens[-1]['probs'][0]
        # Find top prediction that isn't a filtered token
        sorted_indices = torch.argsort(final_probs, descending=True)
        final_top_word = '?'
        for idx in sorted_indices:
            word = idx_to_word.get(idx.item(), '?')
            if word not in self.filter_tokens:
                final_top_word = word
                break
        
        for stage_idx, stage_data in enumerate(logit_lens):
            x_start = 40 + stage_idx * stage_width
            probs = stage_data['probs'][0]
            stage_name = stage_data['stage']
            
            # Get top predictions, filtering out placeholder tokens
            sorted_indices = torch.argsort(probs, descending=True)
            top_items = []
            for idx in sorted_indices:
                word = idx_to_word.get(idx.item(), '?')
                if word not in self.filter_tokens:
                    top_items.append((word, probs[idx].item()))
                    if len(top_items) >= top_k:
                        break
            
            # Stage label
            self.logit_lens_canvas.create_text(
                x_start + stage_width // 2, padding_top,
                text=stage_name, fill='#69b4ff', font=('Consolas', 9, 'bold')
            )
            
            # Draw bars for top predictions
            max_bar_width = stage_width - 20
            
            for i, (word, prob_val) in enumerate(top_items):
                y = padding_top + 20 + i * (bar_height + 8)
                
                bar_width = prob_val * max_bar_width
                
                # Highlight if this is the final answer
                if word == final_top_word:
                    color = '#00ff88'  # Green for final answer
                    text_color = '#00ff88'
                else:
                    # Color gradient based on probability
                    intensity = int(prob_val * 200) + 55
                    color = f'#{intensity:02x}{intensity:02x}00'  # Yellow tones
                    text_color = '#cccccc'
                
                # Draw bar
                self.logit_lens_canvas.create_rectangle(
                    x_start, y, x_start + bar_width, y + bar_height,
                    fill=color, outline='#333'
                )
                
                # Word label
                display_word = word[:7] if len(word) > 7 else word
                self.logit_lens_canvas.create_text(
                    x_start + 3, y + bar_height // 2,
                    text=display_word, fill='black' if prob_val > 0.3 else text_color,
                    font=('Consolas', 8), anchor='w'
                )
                
                # Probability value
                self.logit_lens_canvas.create_text(
                    x_start + max_bar_width, y + bar_height // 2,
                    text=f"{prob_val:.2f}", fill='#888', font=('Consolas', 7), anchor='e'
                )
        
        # Legend
        legend_y = canvas_height - 25
        self.logit_lens_canvas.create_rectangle(10, legend_y, 25, legend_y + 12, fill='#00ff88', outline='#333')
        self.logit_lens_canvas.create_text(30, legend_y + 6, text=f"= Final answer: '{final_top_word}'", 
                                           fill='#00ff88', font=('Consolas', 8), anchor='w')
    
    def update_qkv_inspector(self, layer_idx, head_idx):
        """Visualize Query, Key, Value tensors and attention computation"""
        self.qkv_canvas.delete("all")
        
        if not self.last_layer_data or not self.last_context:
            self.qkv_canvas.create_text(200, 150, text="No data yet", fill='#666')
            return
        
        # Get QKV data from the last generation step
        last_step_data = self.last_layer_data[-1] if self.last_layer_data else None
        if not last_step_data or 'qkv_data' not in last_step_data:
            self.qkv_canvas.create_text(200, 150, text="No Q,K,V data available", fill='#666')
            return
        
        qkv_data = last_step_data['qkv_data'][layer_idx][head_idx]
        if qkv_data['q'] is None:
            self.qkv_canvas.create_text(200, 150, text="No Q,K,V data", fill='#666')
            return
        
        canvas_width = self.qkv_canvas.winfo_width() or 750
        
        # Get full tensors
        Q_full = qkv_data['q'][0].numpy()  # [seq_len, head_dim]
        K_full = qkv_data['k'][0].numpy()
        V_full = qkv_data['v'][0].numpy()
        scores_full = qkv_data['scores'][0].numpy()  # [seq_len, seq_len] pre-softmax
        attn_full = qkv_data['attn'][0].numpy()  # [seq_len, seq_len] post-softmax
        
        seq_len = Q_full.shape[0]
        head_dim = Q_full.shape[1]
        
        # Get context tokens (limited to what the model actually processed)
        context_indices = self.last_context[-seq_len:]
        
        # Filter out placeholder tokens
        filtered_tokens, filtered_indices = self.filter_display_tokens(context_indices)
        
        # Ensure indices are valid for the matrices
        valid_pairs = [(t, i) for t, i in zip(filtered_tokens, filtered_indices) if i < seq_len]
        if valid_pairs:
            filtered_tokens, filtered_indices = zip(*valid_pairs)
            filtered_tokens = list(filtered_tokens)
            filtered_indices = list(filtered_indices)
        else:
            return
        
        # Limit to last N tokens for display
        max_display = 8
        if len(filtered_tokens) > max_display:
            filtered_tokens = filtered_tokens[-max_display:]
            filtered_indices = filtered_indices[-max_display:]
        
        n_tokens = len(filtered_tokens)
        
        if n_tokens == 0:
            return
        
        # Extract filtered rows/columns from matrices
        tokens = filtered_tokens
        Q = Q_full[filtered_indices, :]
        K = K_full[filtered_indices, :]
        V = V_full[filtered_indices, :]
        scores = scores_full[np.ix_(filtered_indices, filtered_indices)]
        attn = attn_full[np.ix_(filtered_indices, filtered_indices)]
        
        padding = 20
        section_gap = 25
        y_pos = padding
        
        # Color function for heatmaps
        def val_to_color(val, max_val, colormap='blue'):
            norm = min(max(val / (max_val + 1e-8), -1), 1)
            if colormap == 'blue':
                if norm >= 0:
                    intensity = int(norm * 200) + 55
                    return f'#5555{intensity:02x}'
                else:
                    intensity = int(-norm * 200) + 55
                    return f'#{intensity:02x}5555'
            elif colormap == 'green':
                if norm >= 0:
                    intensity = int(norm * 200) + 55
                    return f'#55{intensity:02x}55'
                else:
                    intensity = int(-norm * 200) + 55
                    return f'#{intensity:02x}5555'
            elif colormap == 'purple':
                if norm >= 0:
                    intensity = int(norm * 200) + 55
                    return f'#{intensity:02x}55{intensity:02x}'
                else:
                    intensity = int(-norm * 200) + 55
                    return f'#{intensity:02x}5555'
            else:  # attention (yellow-green)
                intensity = int(abs(norm) * 200) + 55
                return f'#{intensity:02x}{intensity:02x}55'
        
        cell_size = min(22, (canvas_width - 200) // max(n_tokens, head_dim))
        
        # =========== SECTION 1: QUERY ===========
        self.qkv_canvas.create_text(
            padding, y_pos, text="QUERY (Q) - \"What am I looking for?\"",
            fill='#69b4ff', font=('Consolas', 10, 'bold'), anchor='w'
        )
        y_pos += 18
        self.qkv_canvas.create_text(
            padding, y_pos, text="Each row is a token's query vector. Similar queries = looking for similar things.",
            fill='#888', font=('Consolas', 8), anchor='w'
        )
        y_pos += 18
        
        # Draw Q heatmap
        max_q = max(abs(Q.max()), abs(Q.min()))
        for i, token in enumerate(tokens):
            # Token label
            display_token = token[:5] if len(token) > 5 else token
            self.qkv_canvas.create_text(
                padding + 45, y_pos + i * cell_size + cell_size // 2,
                text=display_token, fill='#aaa', font=('Consolas', 8), anchor='e'
            )
            # Heatmap cells
            for j in range(min(head_dim, 12)):  # Show first 12 dims
                x = padding + 50 + j * cell_size
                y = y_pos + i * cell_size
                color = val_to_color(Q[i, j], max_q, 'blue')
                self.qkv_canvas.create_rectangle(
                    x, y, x + cell_size - 1, y + cell_size - 1,
                    fill=color, outline='#333'
                )
        
        # Dimension labels
        for j in range(min(head_dim, 12)):
            self.qkv_canvas.create_text(
                padding + 50 + j * cell_size + cell_size // 2, y_pos + n_tokens * cell_size + 8,
                text=str(j), fill='#666', font=('Consolas', 6)
            )
        
        y_pos += n_tokens * cell_size + section_gap
        
        # =========== SECTION 2: KEY ===========
        self.qkv_canvas.create_text(
            padding, y_pos, text="KEY (K) - \"What do I contain/offer?\"",
            fill='#00ff88', font=('Consolas', 10, 'bold'), anchor='w'
        )
        y_pos += 18
        self.qkv_canvas.create_text(
            padding, y_pos, text="Each row is a token's key vector. When Q·K is high, attention flows.",
            fill='#888', font=('Consolas', 8), anchor='w'
        )
        y_pos += 18
        
        # Draw K heatmap
        max_k = max(abs(K.max()), abs(K.min()))
        for i, token in enumerate(tokens):
            display_token = token[:5] if len(token) > 5 else token
            self.qkv_canvas.create_text(
                padding + 45, y_pos + i * cell_size + cell_size // 2,
                text=display_token, fill='#aaa', font=('Consolas', 8), anchor='e'
            )
            for j in range(min(head_dim, 12)):
                x = padding + 50 + j * cell_size
                y = y_pos + i * cell_size
                color = val_to_color(K[i, j], max_k, 'green')
                self.qkv_canvas.create_rectangle(
                    x, y, x + cell_size - 1, y + cell_size - 1,
                    fill=color, outline='#333'
                )
        
        y_pos += n_tokens * cell_size + section_gap
        
        # =========== SECTION 3: VALUE ===========
        self.qkv_canvas.create_text(
            padding, y_pos, text="VALUE (V) - \"What information do I pass along?\"",
            fill='#ff69b4', font=('Consolas', 10, 'bold'), anchor='w'
        )
        y_pos += 18
        self.qkv_canvas.create_text(
            padding, y_pos, text="The actual content that gets passed when this token is attended to.",
            fill='#888', font=('Consolas', 8), anchor='w'
        )
        y_pos += 18
        
        # Draw V heatmap
        max_v = max(abs(V.max()), abs(V.min()))
        for i, token in enumerate(tokens):
            display_token = token[:5] if len(token) > 5 else token
            self.qkv_canvas.create_text(
                padding + 45, y_pos + i * cell_size + cell_size // 2,
                text=display_token, fill='#aaa', font=('Consolas', 8), anchor='e'
            )
            for j in range(min(head_dim, 12)):
                x = padding + 50 + j * cell_size
                y = y_pos + i * cell_size
                color = val_to_color(V[i, j], max_v, 'purple')
                self.qkv_canvas.create_rectangle(
                    x, y, x + cell_size - 1, y + cell_size - 1,
                    fill=color, outline='#333'
                )
        
        y_pos += n_tokens * cell_size + section_gap
        
        # =========== SECTION 4: Q·K Scores (pre-softmax) ===========
        self.qkv_canvas.create_text(
            padding, y_pos, text="Q·K SCORES (pre-softmax) - \"How well does each query match each key?\"",
            fill='#ffaa00', font=('Consolas', 10, 'bold'), anchor='w'
        )
        y_pos += 18
        self.qkv_canvas.create_text(
            padding, y_pos, text="Raw dot products. Higher = stronger match. -inf (masked) for future tokens.",
            fill='#888', font=('Consolas', 8), anchor='w'
        )
        y_pos += 18
        
        # Column headers for score matrix
        for j, token in enumerate(tokens):
            display_token = token[:3] if len(token) > 3 else token
            self.qkv_canvas.create_text(
                padding + 50 + j * cell_size + cell_size // 2, y_pos,
                text=display_token, fill='#aaa', font=('Consolas', 7), angle=0
            )
        y_pos += 15
        
        # Draw score matrix
        finite_scores = scores[~np.isinf(scores)]
        max_score = max(abs(finite_scores.max()), abs(finite_scores.min())) if len(finite_scores) > 0 else 1
        
        for i, token in enumerate(tokens):
            display_token = token[:5] if len(token) > 5 else token
            self.qkv_canvas.create_text(
                padding + 45, y_pos + i * cell_size + cell_size // 2,
                text=display_token, fill='#aaa', font=('Consolas', 8), anchor='e'
            )
            for j in range(n_tokens):
                x = padding + 50 + j * cell_size
                y = y_pos + i * cell_size
                score = scores[i, j]
                
                if np.isinf(score) and score < 0:
                    # Masked (future token)
                    color = '#222'
                    self.qkv_canvas.create_rectangle(
                        x, y, x + cell_size - 1, y + cell_size - 1,
                        fill=color, outline='#333'
                    )
                    self.qkv_canvas.create_line(
                        x + 2, y + 2, x + cell_size - 3, y + cell_size - 3, fill='#444'
                    )
                else:
                    color = val_to_color(score, max_score, 'attention')
                    self.qkv_canvas.create_rectangle(
                        x, y, x + cell_size - 1, y + cell_size - 1,
                        fill=color, outline='#333'
                    )
        
        y_pos += n_tokens * cell_size + section_gap
        
        # =========== SECTION 5: Attention Weights (post-softmax) ===========
        self.qkv_canvas.create_text(
            padding, y_pos, text="ATTENTION WEIGHTS (post-softmax) - \"Final attention distribution\"",
            fill='#00ffff', font=('Consolas', 10, 'bold'), anchor='w'
        )
        y_pos += 18
        self.qkv_canvas.create_text(
            padding, y_pos, text="After softmax, each row sums to 1.0. This weights the Value vectors.",
            fill='#888', font=('Consolas', 8), anchor='w'
        )
        y_pos += 18
        
        # Column headers
        for j, token in enumerate(tokens):
            display_token = token[:3] if len(token) > 3 else token
            self.qkv_canvas.create_text(
                padding + 50 + j * cell_size + cell_size // 2, y_pos,
                text=display_token, fill='#aaa', font=('Consolas', 7)
            )
        y_pos += 15
        
        # Draw attention matrix
        for i, token in enumerate(tokens):
            display_token = token[:5] if len(token) > 5 else token
            self.qkv_canvas.create_text(
                padding + 45, y_pos + i * cell_size + cell_size // 2,
                text=display_token, fill='#aaa', font=('Consolas', 8), anchor='e'
            )
            for j in range(n_tokens):
                x = padding + 50 + j * cell_size
                y = y_pos + i * cell_size
                weight = attn[i, j]
                
                # Brighter = higher attention
                intensity = int(weight * 230) + 25
                color = f'#{intensity:02x}{intensity:02x}{int(intensity*0.6):02x}'
                
                self.qkv_canvas.create_rectangle(
                    x, y, x + cell_size - 1, y + cell_size - 1,
                    fill=color, outline='#333'
                )
                
                # Show value if significant
                if weight > 0.15:
                    text_color = 'black' if intensity > 150 else 'white'
                    self.qkv_canvas.create_text(
                        x + cell_size // 2, y + cell_size // 2,
                        text=f"{weight:.1f}", fill=text_color, font=('Consolas', 6)
                    )
        
        y_pos += n_tokens * cell_size + section_gap
        
        # =========== SECTION 6: Formula Reminder ===========
        self.qkv_canvas.create_text(
            padding, y_pos, text="Formula: Attention(Q,K,V) = softmax(Q·Kᵀ / √d) × V",
            fill='#ffaa00', font=('Consolas', 9, 'bold'), anchor='w'
        )
        y_pos += 18
        self.qkv_canvas.create_text(
            padding, y_pos, 
            text=f"Layer {layer_idx}, Head {head_idx} | d_k = {head_dim} | √d_k = {head_dim**0.5:.2f}",
            fill='#666', font=('Consolas', 8), anchor='w'
        )
        y_pos += 25
        
        # Update scroll region
        self.qkv_canvas.configure(scrollregion=(0, 0, canvas_width, y_pos))
    
    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# =============================================================================
# PART 6: Run
# =============================================================================

if __name__ == "__main__":
    print("\nStarting Enhanced GUI...")
    root = tk.Tk()
    app = EnhancedDiagnosticGUI(root, model)
    root.mainloop()
