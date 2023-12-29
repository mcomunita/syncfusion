"""
taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
# sys.path.insert(0, '.')  # nopep8
from CondFoleyGen.specvqgan.utils import instantiate_from_config

logger = logging.getLogger(__name__)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.tril(torch.ones(config.block_size,
                                     config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = self.attn_drop(att) @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))

        return y, att


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        # x = x + self.attn(self.ln1(x))

        # x is a tuple (x, attention)
        x, _ = x
        res = x
        x = self.ln1(x)
        x, att = self.attn(x)
        x = res + x

        x = x + self.mlp(self.ln2(x))

        return x, att


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked)
        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None, targets=None):
        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector

        if embeddings is not None:  # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)

        # returns only last layer attention
        # giving tuple (x, None) just because Sequential takes a single input but outputs two (x, atttention).
        # att is (B, H, T, T)
        x, att = self.blocks((x, None))
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, att


class GPTClass(GPT):

    def __init__(self, token_embedding_config, GPT_config):
        super().__init__(**GPT_config)
        self.embedder = instantiate_from_config(config=token_embedding_config)

    def forward(self, idx, token):
        token = self.embedder(token)
        # calling forward from super
        return super().forward(idx, embeddings=token)


class GPTFeats(GPT):

    def __init__(self, feat_embedding_config, GPT_config):
        super().__init__(**GPT_config)
        print(">>> GPTFeats init")
        # patching the config by removing the default parameters for Conv1d
        if feat_embedding_config["target"].split('.')[-1] in ['LSTM', 'GRU']:
            for p in ['in_channels', 'out_channels', 'padding', 'kernel_size']:
                if p in feat_embedding_config.params:
                    feat_embedding_config.params.pop(p)
        
        print("instantiating embedder")
        self.embedder = instantiate_from_config(config=feat_embedding_config)
        if isinstance(self.embedder, nn.Linear):
            print('Checkout cond_transformer.configure_optimizers. Make sure not to use decay with Linear')

    def forward(self, idx, feats):
        if isinstance(self.embedder, nn.Linear):
            feats = feats.permute(0, 2, 1)
            feats = self.embedder(feats)
        elif isinstance(self.embedder, (nn.LSTM, nn.GRU)):
            feats = feats.permute(0, 2, 1)
            feats, _ = self.embedder(feats)
        elif isinstance(self.embedder, (nn.Conv1d, nn.Identity)):
            # (B, D', T) <- (B, D, T)
            feats = self.embedder(feats)
            # (B, T, D') <- (B, T, D)
            feats = feats.permute(0, 2, 1)
        else:
            raise NotImplementedError
        # calling forward from super
        return super().forward(idx, embeddings=feats)


class GPTFeatsClass(GPT):

    def __init__(self, feat_embedding_config, token_embedding_config, GPT_config):
        super().__init__(**GPT_config)

        # patching the config by removing the default parameters for Conv1d
        if feat_embedding_config.target.split('.')[-1] in ['LSTM', 'GRU']:
            for p in ['in_channels', 'out_channels', 'padding', 'kernel_size']:
                if p in feat_embedding_config.params:
                    feat_embedding_config.params.pop(p)

        self.feat_embedder = instantiate_from_config(config=feat_embedding_config)
        self.cls_embedder = instantiate_from_config(config=token_embedding_config)

        if isinstance(self.feat_embedder, nn.Linear):
            print('Checkout cond_transformer.configure_optimizers. Make sure not to use decay with Linear')

    def forward(self, idx, feats_token_dict: dict):
        feats = feats_token_dict['feature']
        token = feats_token_dict['target']

        # Features. Output size: (B, T, D')
        if isinstance(self.feat_embedder, nn.Linear):
            feats = feats.permute(0, 2, 1)
            feats = self.feat_embedder(feats)
        elif isinstance(self.feat_embedder, (nn.LSTM, nn.GRU)):
            feats = feats.permute(0, 2, 1)
            feats, _ = self.feat_embedder(feats)
        elif isinstance(self.feat_embedder, (nn.Conv1d, nn.Identity)):
            # (B, D', T) <- (B, D, T)
            feats = self.feat_embedder(feats)
            # (B, T, D') <- (B, T, D)
            feats = feats.permute(0, 2, 1)
        else:
            raise NotImplementedError

        # Class. Output size: (B, 1, D')
        token = self.cls_embedder(token)

        # Concat
        condition_emb = torch.cat([feats, token], dim=1)

        # calling forward from super
        return super().forward(idx, embeddings=condition_emb)


if __name__ == '__main__':
    import torch
    from omegaconf import OmegaConf
    import numpy as np
    from tqdm import tqdm

    device = torch.device('cuda:2')
    torch.cuda.set_device(device)

    cfg = OmegaConf.load('./configs/vggsound_transformer.yaml')

    model = instantiate_from_config(cfg.model.params.transformer_config)
    model = model.to(device)

    mel_num = cfg.data.params.mel_num
    spec_crop_len = cfg.data.params.spec_crop_len
    feat_depth = cfg.data.params.feat_depth
    feat_crop_len = cfg.data.params.feat_crop_len

    gcd = np.gcd(mel_num, spec_crop_len)
    z_idx_size = (2, int(mel_num / gcd) * int(spec_crop_len / gcd))

    for i in tqdm(range(300)):
        z_indices = torch.randint(0, cfg.model.params.transformer_config.params.GPT_config.vocab_size, z_idx_size).to(device)
        c = torch.rand(2, feat_depth, feat_crop_len).to(device)
        logits, loss, att = model(z_indices[:, :-1], feats=c)
