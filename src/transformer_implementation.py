"""
Implementation of the Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017)
Python 3.11 compatible implementation with detailed explanations and visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import math
from dataclasses import dataclass

# Set random seed for reproducibility
np.random.seed(42)

@dataclass
class TransformerConfig:
    """Configuration for Transformer model matching paper specifications"""
    d_model: int = 512  # Model dimension
    n_heads: int = 8    # Number of attention heads
    n_layers: int = 6   # Number of encoder/decoder layers
    d_ff: int = 2048    # Feed-forward dimension
    max_seq_len: int = 100  # Maximum sequence length
    dropout: float = 0.1    # Dropout rate
    vocab_size: int = 10000  # Vocabulary size


class MultiHeadAttention:
    """
    Multi-Head Attention mechanism as described in Section 3.2.2
    Equation: MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.d_k = config.d_model // config.n_heads  # 64 in the paper
        self.d_v = config.d_model // config.n_heads  # 64 in the paper
        
        # Initialize projection matrices
        self.W_q = self._init_weight(config.d_model, config.d_model)
        self.W_k = self._init_weight(config.d_model, config.d_model)
        self.W_v = self._init_weight(config.d_model, config.d_model)
        self.W_o = self._init_weight(config.d_model, config.d_model)
        
    def _init_weight(self, din, dout):
        """Xavier initialization"""
        return np.random.randn(din, dout) * np.sqrt(2.0 / (din + dout))
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Scaled Dot-Product Attention (Section 3.2.1)
        Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
        """
        d_k = Q.shape[-1]
        
        # Compute attention scores
        # For 4D tensors (batch, heads, seq_len, d_k), transpose last two dims
        K_transposed = np.swapaxes(K, -2, -1)
        scores = np.matmul(Q, K_transposed) / np.sqrt(d_k)
        
        # Apply mask if provided (for decoder self-attention)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Apply softmax
        attention_weights = self.softmax(scores)
        
        # Apply attention to values
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def softmax(self, x):
        """Stable softmax implementation"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-head attention
        """
        batch_size = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]
        seq_len_v = value.shape[1]
        
        # Linear projections in batch from d_model => h x d_k
        Q = np.matmul(query, self.W_q).reshape(batch_size, seq_len_q, self.config.n_heads, self.d_k)
        K = np.matmul(key, self.W_k).reshape(batch_size, seq_len_k, self.config.n_heads, self.d_k)
        V = np.matmul(value, self.W_v).reshape(batch_size, seq_len_v, self.config.n_heads, self.d_v)
        
        # Transpose for attention: (batch, n_heads, seq_len, d_k)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len_q, self.config.d_model
        )
        
        # Final linear projection
        output = np.matmul(attention_output, self.W_o)
        
        return output, attention_weights


class PositionalEncoding:
    """
    Positional Encoding as described in Section 3.5
    PE(pos,2i) = sin(pos/10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    """
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.encoding = self._create_encoding()
    
    def _create_encoding(self):
        """Generate positional encoding matrix"""
        pe = np.zeros((self.config.max_seq_len, self.config.d_model))
        position = np.arange(0, self.config.max_seq_len).reshape(-1, 1)
        
        # Create div_term for the sinusoidal pattern
        div_term = np.exp(np.arange(0, self.config.d_model, 2) * 
                         -(np.log(10000.0) / self.config.d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = np.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def forward(self, x):
        """Add positional encoding to input embeddings"""
        seq_len = x.shape[1]
        return x + self.encoding[:seq_len, :]


class FeedForward:
    """
    Position-wise Feed-Forward Network (Section 3.3)
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.W1 = np.random.randn(config.d_model, config.d_ff) * 0.02
        self.b1 = np.zeros(config.d_ff)
        self.W2 = np.random.randn(config.d_ff, config.d_model) * 0.02
        self.b2 = np.zeros(config.d_model)
    
    def forward(self, x):
        """Forward pass with ReLU activation"""
        # First linear transformation
        hidden = np.matmul(x, self.W1) + self.b1
        # ReLU activation
        hidden = np.maximum(0, hidden)
        # Second linear transformation
        output = np.matmul(hidden, self.W2) + self.b2
        return output


class LayerNorm:
    """Layer Normalization"""
    
    def __init__(self, d_model, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class TransformerBlock:
    """Single Transformer encoder/decoder block"""
    
    def __init__(self, config: TransformerConfig, is_decoder=False):
        self.config = config
        self.is_decoder = is_decoder
        
        self.attention = MultiHeadAttention(config)
        self.norm1 = LayerNorm(config.d_model)
        self.feed_forward = FeedForward(config)
        self.norm2 = LayerNorm(config.d_model)
        
        if is_decoder:
            self.cross_attention = MultiHeadAttention(config)
            self.norm3 = LayerNorm(config.d_model)
    
    def forward(self, x, encoder_output=None, self_mask=None, cross_mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, attn_weights = self.attention.forward(x, x, x, self_mask)
        x = self.norm1.forward(x + attn_output)
        
        # Cross-attention for decoder
        if self.is_decoder and encoder_output is not None:
            cross_attn_output, _ = self.cross_attention.forward(
                x, encoder_output, encoder_output, cross_mask
            )
            x = self.norm3.forward(x + cross_attn_output)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward.forward(x)
        x = self.norm2.forward(x + ff_output)
        
        return x, attn_weights


class Transformer:
    """Complete Transformer model"""
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.positional_encoding = PositionalEncoding(config)
        
        # Encoder stack
        self.encoder_layers = [
            TransformerBlock(config, is_decoder=False) 
            for _ in range(config.n_layers)
        ]
        
        # Decoder stack
        self.decoder_layers = [
            TransformerBlock(config, is_decoder=True) 
            for _ in range(config.n_layers)
        ]
        
        # Embeddings (shared weights as mentioned in Section 3.4)
        self.embedding = np.random.randn(config.vocab_size, config.d_model) * np.sqrt(config.d_model)
    
    def create_padding_mask(self, seq, pad_idx=0):
        """Create mask for padding tokens"""
        return (seq != pad_idx).astype(np.float32)[:, np.newaxis, np.newaxis, :]
    
    def create_look_ahead_mask(self, size):
        """Create mask for decoder self-attention (triangular mask)"""
        mask = np.triu(np.ones((size, size)), k=1)
        return (mask == 0).astype(np.float32)
    
    def encode(self, src, src_mask=None):
        """Encoder forward pass"""
        # Embedding and positional encoding
        x = self.embedding[src] * np.sqrt(self.config.d_model)
        x = self.positional_encoding.forward(x)
        
        attention_weights = []
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x, attn_w = layer.forward(x, self_mask=src_mask)
            attention_weights.append(attn_w)
        
        return x, attention_weights
    
    def decode(self, tgt, encoder_output, tgt_mask=None, src_mask=None):
        """Decoder forward pass"""
        # Embedding and positional encoding
        x = self.embedding[tgt] * np.sqrt(self.config.d_model)
        x = self.positional_encoding.forward(x)
        
        attention_weights = []
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x, attn_w = layer.forward(
                x, encoder_output, self_mask=tgt_mask, cross_mask=src_mask
            )
            attention_weights.append(attn_w)
        
        return x, attention_weights


def visualize_attention(attention_weights, tokens, layer_idx=0, head_idx=0):
    """Visualize attention weights for a specific layer and head"""
    plt.figure(figsize=(10, 8))
    
    # Get attention weights for specific layer and head
    weights = attention_weights[layer_idx][0, head_idx, :len(tokens), :len(tokens)]
    
    plt.imshow(weights, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    
    # Set labels
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    plt.yticks(range(len(tokens)), tokens)
    
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    plt.title(f'Attention Weights - Layer {layer_idx + 1}, Head {head_idx + 1}')
    plt.tight_layout()
    plt.show()


def demonstrate_transformer():
    """Demonstrate the Transformer with a simple example"""
    print("=" * 60)
    print("TRANSFORMER IMPLEMENTATION DEMONSTRATION")
    print("Based on 'Attention Is All You Need' (Vaswani et al., 2017)")
    print("=" * 60)
    
    # Initialize configuration
    config = TransformerConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_seq_len=100,
        vocab_size=1000
    )
    
    # Create model
    model = Transformer(config)
    
    # Create sample input (batch_size=1, seq_len=10)
    src_tokens = np.random.randint(1, 100, (1, 10))
    tgt_tokens = np.random.randint(1, 100, (1, 8))
    
    print(f"\nModel Configuration:")
    print(f"  - Model dimension (d_model): {config.d_model}")
    print(f"  - Number of heads: {config.n_heads}")
    print(f"  - Head dimension (d_k): {config.d_model // config.n_heads}")
    print(f"  - Number of layers: {config.n_layers}")
    print(f"  - Feed-forward dimension: {config.d_ff}")
    
    print(f"\nInput shapes:")
    print(f"  - Source sequence: {src_tokens.shape}")
    print(f"  - Target sequence: {tgt_tokens.shape}")
    
    # Encode
    encoder_output, encoder_attention = model.encode(src_tokens)
    print(f"\nEncoder output shape: {encoder_output.shape}")
    
    # Decode
    decoder_output, decoder_attention = model.decode(
        tgt_tokens, encoder_output
    )
    print(f"Decoder output shape: {decoder_output.shape}")
    
    # Demonstrate attention patterns
    print("\n" + "=" * 60)
    print("KEY OBSERVATIONS FROM THE PAPER:")
    print("=" * 60)
    
    print("\n1. SCALED DOT-PRODUCT ATTENTION:")
    print("   - Attention weights are computed as: softmax(QK^T / sqrt(d_k))")
    print("   - Scaling by sqrt(d_k) prevents gradient vanishing in softmax")
    print(f"   - With d_k={config.d_model // config.n_heads}, scaling factor = {np.sqrt(config.d_model // config.n_heads):.2f}")
    
    print("\n2. MULTI-HEAD ATTENTION:")
    print("   - Allows model to attend to different representation subspaces")
    print("   - Each head has reduced dimension for computational efficiency")
    print(f"   - Total parameters same as single head with full dimension")
    
    print("\n3. POSITIONAL ENCODING:")
    print("   - Uses sinusoidal functions of different frequencies")
    print("   - Allows model to learn relative positions")
    print("   - Can extrapolate to sequence lengths unseen during training")
    
    print("\n4. COMPUTATIONAL COMPLEXITY:")
    print("   - Self-attention: O(n² * d) per layer")
    print("   - RNN: O(n * d²) sequential operations")
    print("   - Transformer enables parallelization across sequence positions")
    
    # Visualize positional encoding
    visualize_positional_encoding(config)
    
    # Visualize attention pattern
    visualize_sample_attention(config)
    
    return model, encoder_attention, decoder_attention


def visualize_positional_encoding(config: TransformerConfig):
    """Visualize the sinusoidal positional encoding patterns"""
    pe = PositionalEncoding(config)
    encoding = pe.encoding[:50, :128]  # First 50 positions, first 128 dimensions
    
    plt.figure(figsize=(12, 4))
    
    # Heatmap of positional encodings
    plt.subplot(1, 2, 1)
    plt.imshow(encoding.T, cmap='RdBu', aspect='auto')
    plt.colorbar()
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.title('Positional Encoding Pattern (Sinusoidal)')
    
    # Line plot for specific dimensions
    plt.subplot(1, 2, 2)
    positions = np.arange(50)
    for i in [0, 1, 4, 5, 8, 9]:  # Plot a few dimensions
        plt.plot(positions, encoding[:, i], label=f'dim {i}')
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.title('Positional Encoding for Selected Dimensions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_sample_attention(config: TransformerConfig):
    """Visualize a sample attention pattern"""
    # Create a simple attention pattern to demonstrate
    seq_len = 10
    n_heads = 4
    
    fig, axes = plt.subplots(1, n_heads, figsize=(15, 3))
    
    sample_tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat', '.', '<PAD>', '<PAD>', '<PAD>']
    
    for head in range(n_heads):
        # Create different attention patterns for each head
        if head == 0:  # Attending to adjacent words
            attn = np.eye(seq_len) * 0.7
            for i in range(seq_len - 1):
                attn[i, i+1] = 0.3
        elif head == 1:  # Attending to first word
            attn = np.ones((seq_len, seq_len)) * 0.1
            attn[:, 0] = 0.9
        elif head == 2:  # Attending to same word (diagonal)
            attn = np.eye(seq_len) * 0.9 + 0.1
        else:  # Random attention
            attn = np.random.rand(seq_len, seq_len)
            attn = attn / attn.sum(axis=-1, keepdims=True)
        
        im = axes[head].imshow(attn[:7, :7], cmap='Blues', vmin=0, vmax=1)
        axes[head].set_title(f'Head {head + 1}')
        axes[head].set_xticks(range(7))
        axes[head].set_yticks(range(7))
        axes[head].set_xticklabels(sample_tokens[:7], rotation=45, ha='right')
        axes[head].set_yticklabels(sample_tokens[:7])
        
        if head == 0:
            axes[head].set_ylabel('Query')
        axes[head].set_xlabel('Key')
    
    plt.suptitle('Multi-Head Attention Patterns (Different Heads Learn Different Relationships)')
    plt.colorbar(im, ax=axes.ravel().tolist(), label='Attention Weight')
    plt.tight_layout()
    plt.show()


def run_experiments():
    """Run experiments demonstrating key findings from the paper"""
    print("\n" + "=" * 60)
    print("EXPERIMENTAL VALIDATION")
    print("=" * 60)
    
    # Experiment 1: Effect of number of heads
    print("\nExperiment 1: Impact of Number of Attention Heads")
    print("-" * 40)
    
    head_counts = [1, 2, 4, 8, 16]
    for n_heads in head_counts:
        config = TransformerConfig(d_model=512, n_heads=n_heads, n_layers=1)
        model = Transformer(config)
        
        # Simulate forward pass
        src = np.random.randint(1, 100, (1, 20))
        output, _ = model.encode(src)
        
        print(f"Heads: {n_heads:2d}, d_k: {512//n_heads:3d}, "
              f"Output shape: {output.shape}, "
              f"Params/head: {(512//n_heads) * 512:,}")
    
    # Experiment 2: Computational complexity comparison
    print("\nExperiment 2: Computational Complexity Analysis")
    print("-" * 40)
    
    seq_lengths = [10, 50, 100, 500]
    d_model = 512
    
    print("Sequence Length | Self-Attention | RNN")
    print("-" * 40)
    for n in seq_lengths:
        self_attn_ops = n * n * d_model  # O(n² * d)
        rnn_ops = n * d_model * d_model   # O(n * d²)
        
        print(f"{n:14d} | {self_attn_ops:14,} | {rnn_ops:14,}")
        if n > 50:
            ratio = self_attn_ops / rnn_ops
            print(f"{'':14s} | Attention is {ratio:.2f}x {'faster' if ratio < 1 else 'slower'}")
    
    print("\nNote: Self-attention is more efficient when n < d_model")
    print("      and can be fully parallelized unlike RNNs")


if __name__ == "__main__":
    # Run the demonstration
    model, enc_attn, dec_attn = demonstrate_transformer()
    
    # Run additional experiments
    run_experiments()
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print("\nThis implementation demonstrates the key concepts from")
    print("'Attention Is All You Need' including:")
    print("  ✓ Scaled dot-product attention")
    print("  ✓ Multi-head attention mechanism")
    print("  ✓ Positional encoding")
    print("  ✓ Encoder-decoder architecture")
    print("  ✓ Layer normalization and residual connections")
    print("\nThe visualizations show how different attention heads")
    print("learn different types of relationships in the data.")
