## Standard libraries
import os
from typing import Any
from functools import partial
from typing import Any

## JAX
import jax
import jax.numpy as jnp
from jax import random
from jax import Array, lax

## Flax 
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints

## Optax (Optimizers in JAX)
import optax #cosine annealing + adam?

## PyTorch
import torch
import torch.utils.data as data
#from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR100




#traditional formual
def scaled_dot_product(k, q, v, mask=None): 
    d_k = k.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(d_k) #q_first_dim, k_last_dim
    #print("shape of attn_logits", attn_logits.shape)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
    attention = nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values, attention

#assume mask.ndim is minimum of 3
def expand_mask(mask):
    assert mask.ndim >= 2
    if mask.ndim == 3:
        mask = jnp.expand_dims(mask, 1)
      #  mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = jnp.expand_dims(mask, 0)
        # mask = mask.unsqueeze(0)
    return mask

#reference: https://github.com/VachanVY/Rotary-Embeddings/blob/main/rope.py  
class PositionalEmbedding:
    def __init__(self, max_len: int, d_model: int):
        p, i = jnp.meshgrid(jnp.arange(float(max_len)),jnp.arange(d_model/2) * 2)
        #each shape of (d_model / 2, max_len)
        theta = (p/1e4**(i/d_model)).T  #(d_model/2, max_len)
        self.pos_emb = jnp.stack([
            jnp.sin(theta), jnp.cos(theta)], axis = -1
        ) #shape (max_len, d_model/2, 2)
        self.pos_emb = self.pos_emb.reshape((max_len, d_model))[None] #(1, max_len, d_model)
    def sin_cos_embs(self):
        return self.pos_emb #(1, max_len, d_model)
    
    def compute_freqs(self):
        sin_freqs = jnp.repeat(self.pos_emb
            [..., None, ::2], repeats = 2, axis=-1
        ) # (1, max_len, 1, d_model)
        cos_freqs = jnp.repeat(self.pos_emb
            [..., None, 1::2], repeats = 2, axis=-1  
                            )
        return sin_freqs, cos_freqs
   
#my implementation... 
def apply_rotary_embeddings(q:Array, k:Array, sin_freqs:Array, cos_freqs:Array):
    T = q.shape[1]
    #shape (batch_dim, num_heads, d_model, d_model / num_heads)
    minus_swap_alternate = lambda x: jnp.stack([-x[..., 1::2], x[..., ::2]], axis=-1).reshape(x.shape)
    q = q*cos_freqs[:, :T, :, :] + minus_swap_alternate(q)*sin_freqs[:, :T, :, :] # (B, T, h, dq)*(1, T, 1, dq) + (B, T, h, dq)*(1, T, 1, dq)
    k = k*cos_freqs[:, :T, :, :] + minus_swap_alternate(k)*sin_freqs[:, :T, :, :] # (B, T, h, dq)*(1, T, 1, dq) + (B, T, h, dq)*(1, T, 1, dq)
    return q, k # (B, T, h, dq), (B, T, h, dq)

#try RoPE precomputed embedding implementation as helper, no state management wrapper 
#compute sin and cos for each position 
@nn.nowrap
def precompute_sin_cos_exp_caches(max_len, d_model):
    #position inddices, [0.... seq_len -1]
    position = jnp.arange(0, max_len, dtype=jnp.float32)[:, None]
    #10000 ^ (-2(i-1) / d_model) for i from 1 to d_model/2
    denominator = 1. / jnp.power(
            10000.0 ,
            jnp.arange(0, d_model, 2) / d_model
        )
    #product of position and denominator
    idx_pos_denom = position * denominator
    #directly concatenate so that there is row of pos_denom_product for sin and cos individually
    idx_concat = jnp.concat([idx_pos_denom, idx_pos_denom], axis=-1)

    assert idx_concat.shape == (max_len, d_model)
    #expand middle two dimensions
    cos_cache = jnp.cos(idx_concat)
    sin_cache = jnp.sin(idx_concat)
    return sin_cache, cos_cache

@nn.nowrap
#function to apply ROPE on a key or query 
def rope_emb( key, query, sin_cache, cos_cache, expand_dim = 1):
    #shape: [num_heads, 1, head_dim], so first and third dimension should match key and query dimensions to broadcast
    cos_cache = jnp.expand_dims(cos_cache, axis = expand_dim)
    sin_cache = jnp.expand_dims(sin_cache, axis = expand_dim)
    assert (sin_cache.shape == cos_cache.shape)
    #split query and k into two partsn and rotate, with rotational negation of second half
    query1, query2 = jnp.split(query, 2, axis = -1)
    #[batch_size, num_heads, seq_len, head_dim]
    query_rotated = jnp.concatenate([-query2, query1], axis = -1)
    key1, key2 = jnp.split(key, 2, axis = -1)
    key_rotated = jnp.concatenate([-key2, key1], axis = -1)
    
    #apply embs to query and key splits
    query_emb = (query * cos_cache) + (query_rotated * sin_cache)
    key_emb = (key * cos_cache) + (key_rotated* sin_cache)
    return query_emb, key_emb


#multi head latent attention
class MLA(nn.Module):
    num_heads: int
    embed_dim: int
    
    def setup(self):
        self.head_dim = self.embed_dim // self.num_heads #since should naturally project below onto oringal embed_dim
        self.query_proj = nn.Dense(
            self.num_heads * self.head_dim,
            kernel_init = nn.initializers.xavier_uniform()
        )
        self.key_proj = nn.Dense(
            #normally would project ot a reduced number of key-value heads..
            self.num_heads * self.head_dim,
            kernel_init = nn.initializers.xavier_uniform()
        )
        self.value_proj = nn.Dense(
            self.num_heads * self.head_dim,
            kernel_init=nn.initializers.xavier_uniform()
        )
    
        #final output projection of attention
        self.o_proj = nn.Dense(
            self.embed_dim,
            kernel_init = nn.initializers.xavier_uniform()
        )
    def __call__(self, x,
                sin, #position embeddings...
                cos, 
                mask=None):
        #get these for scaling reshapes below
        batch_size, seq_length, embed_dim = x.shape

        if mask is not None: 
            #mask min 3 dims
            mask = expand_mask(mask) 
            
        #up projection
        #shape [batch_size, seq_length, num_heads*head_dim]
        query_state = self.query_proj(x)
        key_state = self.key_proj(x)
        value_state = self.value_proj(x)
        
        #[batch_size, seq_length, num_heads*head_dim] ->
        #[batch_size, seq_length, num_heads, head_dim] ->
        #[batch_size, num_heads, seq_length, head_dim]
        query_state = query_state.reshape(
            batch_size, seq_length, self.num_heads, -1
        ).transpose(0, 2, 1, 3)
        key_state = key_state.reshape(
            batch_size, seq_length, self.num_heads, -1
        ).transpose(0, 2, 1, 3)
        value_state = value_state.reshape(
            batch_size, seq_length, self.num_heads, -1
        ).transpose(0, 2, 1, 3)
        
        #applied rope embeddings (shape maintained)
        #TODO Change rotary emb function if needed
        query_state, key_state = apply_rotary_embeddings(query_state, 
                                        key_state, 
                                        sin, 
                                        cos)
        
        #scaled dot product attention with mask
        values, attention = scaled_dot_product(key_state, query_state, value_state, mask)
        
        #softmax to get attention probabilities [batch_size, num_heads, seq_length, head_dim / 2]
        attention_weights = nn.softmax(attention, axis = -1)
        #dot product of weights with values
        attn_output = jnp.matmul(attention_weights, values)
        #reshape output, compress last two dimensions for each head
        #-> [batch_size, num_heads, embed_dim]
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim)
        #attn_output proj back to original shape of x: [batch_size, seq_length, embed_dim]
        #attention_weights maintains k q v state shapes: [batch_size, num_heads, seq_length, head_dim ]
        return self.o_proj(attn_output), attention_weights
    

class MLAEncoderBlock(nn.Module): 
    input_dim : int #determines out dim of mha too
    num_heads : int
    dim_feedforward : int
    dropout_rate : float = 0.0 
    
    def setup(self): 
        self.mla = MLA(self.num_heads, self.input_dim) 
        #MLP
        self.linear = [
            nn.Dense(self.dim_feedforward),
            #is it alright to have deterministic set here in linear, or must it be in forward?
            nn.Dropout(self.dropout_rate),
            nn.relu,
            nn.Dense(self.input_dim)
        ]
        #norm1 and norm2 initialized as separate layers otherwise will share weights 
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def __call__(self, x, sin, cos, 
                mask=None, deterministic = False):
        attn_, _ =self.mla(x,  sin, cos, mask)

        #add & norm after mha. Dropout applies during train 
        
        x = x + self.dropout(attn_, deterministic = deterministic)
        x = self.norm1(x)
        
        #x = LayerNorm(x + MLP(x)) so feedforward -> add & norm1
        linear_out = x
        for l in self.linear:
            #apply deterministic where necessary
            linear_out = l(linear_out, deterministic = deterministic) if isinstance(l, nn.Dropout) else l(linear_out)
        x = x + self.dropout(linear_out, deterministic = deterministic)
        x = self.norm2(x)
        return x


#encoder for num_layers encoder blocks
class MLATransformerEncoder(nn.Module): 
    num_layers : int #num encoder layers in Encoder block
    input_dim : int #determines out dim of mha too
    num_heads : int
    dim_feedforward : int
    dropout_rate : float = 0.0
    
    def setup(self):
        #initializes train parameter from self.train in the individual encoder blocks
        self.layers = [MLAEncoderBlock(
            input_dim = self.input_dim, 
            num_heads = self.num_heads, 
            dim_feedforward = self.dim_feedforward,
            dropout_rate = self.dropout_rate
        )
        for i in range(self.num_layers)]
        #pass dynamic deterministic variable as call argument
    def __call__(self, x, 
                sin, #precomputed from precompute_sin_cos_exp_caches()
                cos, 
                mask=None, deterministic = False):
        for layer in self.layers:
            x = layer(x, sin, cos, mask, deterministic = deterministic)
        return x
    
    

#encoder-only full model with multi-head latent attention
class MLATransformer(nn.Module): 
    model_dim : int
    num_heads : int
    num_classes : int
    num_layers : int
    dropout_rate : float = 0.0
    init_dropout_rate : float = 0.0
    
    def setup(self):
        #dropout and output model_dim size
        self.input_dropout = nn.Dropout(rate = self.init_dropout_rate)
        self.input_layer = nn.Dense(self.model_dim)

        self.enc = MLATransformerEncoder(
            num_layers = self.num_layers,
            input_dim = self.model_dim,
            num_heads = self.num_heads,
            dim_feedforward = 2 * self.model_dim,
            dropout_rate = self.dropout_rate
        )
        
        #mlp output (no decoder for now)
        self.out = [
            nn.Dense(self.model_dim),
            nn.LayerNorm(),
            nn.relu,
            nn.Dropout(rate = self.dropout_rate),
            nn.Dense(self.num_classes)
        ]
    
    def __call__(self, 
            x,
            sin,
            cos,
            mask=None,
            train = True):
        x = self.input_dropout(x, deterministic = not train)
        x = self.input_layer(x)
        
        x = self.enc(x, sin, cos, mask, deterministic = not train)
        for l in self.out:
            x = l(x, deterministic = not train) if isinstance(l, nn.Dropout) else l(x)
        return x

