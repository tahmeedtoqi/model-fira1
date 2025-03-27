"""This the new Fira model with MoE Layer. i havnt tested it yet. but it should work fine.
its more advance then the previous one. it has the MoE layer and the transformer block with MoE.
it also has the generation routine and the optimizer configuration. it also has the pretrained weights loading"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

# ----------------------------
# Helper: Weight Initialization
# ----------------------------
def init_linear(linear: nn.Linear, activation: str = 'linear'):
    if activation == 'linear':
        nn.init.xavier_uniform_(linear.weight)
    elif activation == 'relu':
        nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)

# ----------------------------
# Multi‑Head Self‑Attention Module
# ----------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        init_linear(self.query)
        init_linear(self.key)
        init_linear(self.value)
        init_linear(self.out_proj)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        # Linear projections and reshape for multi-head attention.
        q = self.query(x).view(batch_size, seq_len, self.n_head, self.d_head).transpose(1,2)
        k = self.key(x).view(batch_size, seq_len, self.n_head, self.d_head).transpose(1,2)
        v = self.value(x).view(batch_size, seq_len, self.n_head, self.d_head).transpose(1,2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1,2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(context)

# ----------------------------
# Mixture‑of‑Experts (MoE) Layer
# ----------------------------
class MoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        # Create a list of expert feed‑forward networks.
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        for expert in self.experts:
            init_linear(expert[0], activation='relu')
            init_linear(expert[2])
        # A simple gating network: for each token, output logits over experts.
        self.gate = nn.Linear(d_model, num_experts)
        init_linear(self.gate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(x)                  # (batch, seq_len, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)   # (batch, seq_len, num_experts)
        # Compute outputs from each expert and stack them.
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)  # (batch, seq_len, num_experts, d_model)
        gate_probs = gate_probs.unsqueeze(-1)       # (batch, seq_len, num_experts, 1)
        output = torch.sum(gate_probs * expert_outputs, dim=2)  # (batch, seq_len, d_model)
        return output

# ----------------------------
# Transformer Block with MoE
# ----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, num_experts: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = MoE(d_model, d_ff, num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self‑attention sublayer with residual connection.
        attn_out = self.attn(self.ln1(x), mask)
        x = x + self.dropout(attn_out)
        # MoE feed‑forward sublayer with residual connection.
        moe_out = self.moe(self.ln2(x))
        x = x + self.dropout(moe_out)
        return x

# ----------------------------
# FIRA LM Head Model with MoE and Extended Utilities
# ----------------------------
class FIRA(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 768,
                 n_head: int = 12,
                 num_layers: int = 12,
                 d_ff: int = 3072,
                 num_experts: int = 4,
                 max_seq_len: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Token and position embeddings.
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        # Stack transformer blocks.
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, num_experts, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        init_linear(self.lm_head)

        # Store configuration for utilities like MFU estimation.
        self.config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_head": n_head,
            "num_layers": num_layers,
            "d_ff": d_ff,
            "num_experts": num_experts,
            "max_seq_len": max_seq_len,
            "dropout": dropout
        }

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        # Create position indices and sum with token embeddings.
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    # ----------------------------
    # Generation Routine
    # ----------------------------
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """
        Autoregressively generate tokens.
        Args:
            input_ids: Tensor of shape (batch, seq_len) containing input token indices.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature.
            top_k: If provided, sample only from the top_k tokens.
        Returns:
            Tensor of shape (batch, seq_len + max_new_tokens) containing the generated sequence.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # If sequence is too long, crop to max_seq_len.
            if input_ids.size(1) > self.config["max_seq_len"]:
                input_ids = input_ids[:, -self.config["max_seq_len"]:]
            seq_len = input_ids.size(1)
            # Create a causal (lower triangular) mask.
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=input_ids.device)).unsqueeze(0).unsqueeze(0)
            logits = self.forward(input_ids, mask=causal_mask)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

    # ----------------------------
    # Utility: Count Parameters
    # ----------------------------
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Returns the total number of parameters. If non_embedding is True, subtract the position embedding parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embedding.weight.numel()
        return n_params

    # ----------------------------
    # Utility: Estimate MFU
    # ----------------------------
    def estimate_mfu(self, fwdbwd_per_iter: float, dt: float) -> float:
        """
        Estimate model FLOPs utilization (MFU) relative to an A100 GPU's bfloat16 peak (312 TFLOPS).
        Args:
            fwdbwd_per_iter: Number of forward+backward passes per iteration.
            dt: Duration of one iteration in seconds.
        Returns:
            Estimated MFU as a ratio.
        """
        N = self.get_num_params()
        cfg = self.config
        L = cfg["num_layers"]
        H = cfg["n_head"]
        Q = cfg["d_model"] // cfg["n_head"]
        T = cfg["max_seq_len"]
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter / dt
        flops_promised = 312e12  # A100 bfloat16 peak FLOPs
        mfu = flops_achieved / flops_promised
        return mfu

    # ----------------------------
    # Optimizer Configuration
    # ----------------------------
    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: tuple, device_type: str):
        """
        Configure and return an AdamW optimizer with separate parameter groups for weight-decayed and non-decayed parameters.
        Args:
            weight_decay: Weight decay factor.
            learning_rate: Learning rate.
            betas: Tuple of beta coefficients.
            device_type: String indicating the device type ('cuda' or 'cpu').
        Returns:
            An AdamW optimizer instance.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = {'fused': True} if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    # ----------------------------
    # Pretrained Weights Loading
    # ----------------------------
    """unless there is a trained model for testing purpose i have used the GPT2LMHeadModel from Hugging Face. itshould be 
    be replace by our own model"""
    @classmethod
    def from_pretrained(cls, model_type: str, num_experts: int = 4, override_args: dict = None):
        """
        Load weights from a Hugging Face pretrained GPT2LMHeadModel checkpoint.
        Since our model uses a MoE feed‑forward layer, the pretrained MLP weights are copied to the first expert,
        while the remaining experts and the gating network remain randomly initialized.
        Args:
            model_type: A string identifier (e.g., "gpt2", "gpt2-medium", etc.).
            num_experts: Number of experts for the MoE layers.
            override_args: A dictionary to override certain configuration parameters (e.g., dropout).
        Returns:
            An instance of GPT2LMHeadModel with weights loaded where applicable.
        """
        from transformers import GPT2LMHeadModel as HF_GPT2LMHeadModel
        hf_model = HF_GPT2LMHeadModel.from_pretrained(model_type)
        hf_config = hf_model.config
        override_args = override_args or {}
        d_model = hf_config.n_embd
        n_head = hf_config.n_head
        num_layers = hf_config.n_layer
        d_ff = 4 * d_model  # GPT-2 uses 4*d_model for MLP
        max_seq_len = hf_config.n_positions if hasattr(hf_config, 'n_positions') else 1024
        dropout = override_args.get('dropout', hf_config.resid_pdrop if hasattr(hf_config, 'resid_pdrop') else 0.1)
        vocab_size = hf_config.vocab_size

        # Instantiate our model.
        model = cls(vocab_size=vocab_size,
                    d_model=d_model,
                    n_head=n_head,
                    num_layers=num_layers,
                    d_ff=d_ff,
                    num_experts=num_experts,
                    max_seq_len=max_seq_len,
                    dropout=dropout)

        # Copy weights from the Hugging Face model.
        with torch.no_grad():
            model.token_embedding.weight.copy_(hf_model.transformer.wte.weight)
            model.position_embedding.weight.copy_(hf_model.transformer.wpe.weight[:max_seq_len])
            for i, layer in enumerate(model.layers):
                hf_layer = hf_model.transformer.h[i]
                # Copy LayerNorm parameters.
                layer.ln1.weight.copy_(hf_layer.ln_1.weight)
                layer.ln1.bias.copy_(hf_layer.ln_1.bias)
                layer.ln2.weight.copy_(hf_layer.ln_2.weight)
                layer.ln2.bias.copy_(hf_layer.ln_2.bias)
                # Copy self‑attention weights.
                c_attn_weight = hf_layer.attn.c_attn.weight
                c_attn_bias = hf_layer.attn.c_attn.bias
                d_model = model.d_model
                layer.attn.query.weight.copy_(c_attn_weight[:d_model, :])
                layer.attn.query.bias.copy_(c_attn_bias[:d_model])
                layer.attn.key.weight.copy_(c_attn_weight[d_model:2*d_model, :])
                layer.attn.key.bias.copy_(c_attn_bias[d_model:2*d_model])
                layer.attn.value.weight.copy_(c_attn_weight[2*d_model:, :])
                layer.attn.value.bias.copy_(c_attn_bias[2*d_model:])
                layer.attn.out_proj.weight.copy_(hf_layer.attn.c_proj.weight)
                layer.attn.out_proj.bias.copy_(hf_layer.attn.c_proj.bias)
                # For the feed‑forward, copy pretrained weights to the first expert.
                first_expert = layer.moe.experts[0]
                first_expert[0].weight.copy_(hf_layer.mlp.c_fc.weight)
                first_expert[0].bias.copy_(hf_layer.mlp.c_fc.bias)
                first_expert[2].weight.copy_(hf_layer.mlp.c_proj.weight)
                first_expert[2].bias.copy_(hf_layer.mlp.c_proj.bias)
            model.ln_f.weight.copy_(hf_model.transformer.ln_f.weight)
            model.ln_f.bias.copy_(hf_model.transformer.ln_f.bias)
            model.lm_head.weight.copy_(hf_model.lm_head.weight)
        return model

# ----------------------------
# Example Usage
# ----------------------------
if __name__ == '__main__':
    # Example production parameters.
    vocab_size = 50257  # GPT‑2 vocabulary size.
    batch_size = 4
    seq_length = 128

    # Instantiate the model.
    """here you can see i have used the gpt2 params thats because i dont have a trained model.
    and beside i didnt have the time and gpu to go for higher params"""
    model = FIRA(
        vocab_size=vocab_size,
        d_model=768,
        n_head=12,
        num_layers=12,
        d_ff=3072,
        num_experts=4,
        max_seq_len=1024,
        dropout=0.1
    )
    model.eval()

    # Test generation.
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    generated = model.generate(input_ids, max_new_tokens=20, temperature=1.0, top_k=50)
    print("Generated sequence shape:", generated.shape)

    # Estimate MFU (dummy example).
    mfu = model.estimate_mfu(fwdbwd_per_iter=1, dt=0.5)
    print("Estimated MFU:", mfu)

    # Configure optimizer (example for CUDA).
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=1e-4, betas=(0.9, 0.95), device_type='cuda')
    print("Optimizer configured:", optimizer)
