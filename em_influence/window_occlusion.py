"""Windowed occlusion for any HF decoder that routes attention through
ALL_ATTENTION_FUNCTIONS (Llama, Qwen, OLMo-2/3, ...).

For every position t, approximates L(x with t's embedding zeroed) - L(x):
positions t..t+w-1 are recomputed exactly (they attend to the original keys and
values before t, which zeroing t doesn't change); later positions see the new
keys/values of the window, and the resulting change in each layer's attention
output is computed in closed form and taken to the loss with the gradient at
that output. w >= T is exact.
"""

from __future__ import annotations

import torch
from torch.func import functional_call, grad
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class _Probe:
    mode = None
    stash = None
    delta = None
    ts = None
    window_kv = None


PROBE = _Probe()


def probe_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, sliding_window=None, **kwargs):
    i = module.layer_idx
    k = key.repeat_interleave(module.num_key_value_groups, 1)
    v = value.repeat_interleave(module.num_key_value_groups, 1)
    if PROBE.mode == "record":
        T = query.shape[2]
        if sliding_window is not None and T > sliding_window:
            raise ValueError("documents longer than the sliding window aren't supported")
        causal = torch.ones(T, T, dtype=torch.bool, device=query.device).tril()
        a = (query @ k.transpose(-1, -2) * scaling).masked_fill(~causal, float("-inf"))
        o = a.softmax(-1) @ v
        PROBE.stash[i] = dict(q=query * scaling, k=k, v=v, logits=a, o=o)
        if PROBE.delta is not None:
            o = o + PROBE.delta[i]
        return o.transpose(1, 2), None
    # window: query [n,H,w,D] for copies starting at PROBE.ts
    c = PROBE.stash[i]
    ts = PROBE.ts
    n, _, w, _ = query.shape
    T = c["k"].shape[2]
    dev = query.device
    valid = (ts[:, None] + torch.arange(w, device=dev)[None]) < T
    prefix = torch.arange(T, device=dev)[None, :] < ts[:, None]
    causal = torch.ones(w, w, dtype=torch.bool, device=dev).tril()[None] & valid[:, None, :]
    a1 = torch.einsum("nhjd,hsd->nhjs", query * scaling, c["k"][0])
    a1 = a1.masked_fill(~prefix[:, None, None, :], float("-inf"))
    a2 = (query @ k.transpose(-1, -2) * scaling).masked_fill(~causal[:, None], float("-inf"))
    p = torch.cat([a1, a2], -1).softmax(-1)
    o = torch.einsum("nhjs,hsd->nhjd", p[..., :T], c["v"][0]) + p[..., T:] @ v
    PROBE.window_kv[i] = (k, v)
    return o.transpose(1, 2), None


ALL_ATTENTION_FUNCTIONS.register("occlusion_probe", probe_attention)


def _downstream(stash, dO, window_kv, ts, w):
    """sum over layers, heads and queries s >= t+w of dO_s . (change in s's
    attention output when the window's keys/values replace the originals)."""
    total = 0
    for i in sorted(stash):
        c = stash[i]
        k2, v2 = window_kv[i]  # [n,H,w,D]
        a, q, o, v = c["logits"][0], c["q"][0], c["o"][0], c["v"][0]
        T = a.shape[-1]
        dev = a.device
        pos = ts[:, None] + torch.arange(w, device=dev)[None]
        valid = pos < T
        posc = pos.clamp(max=T - 1)
        down = torch.arange(T, device=dev)[None, :] >= (ts[:, None] + w)  # [n, s]
        g = dO[i][0]
        lse = torch.logsumexp(a, -1)
        go = (g * o).sum(-1)
        m = valid[:, None, None, :] & down[:, None, :, None]  # [n,1,S,w]
        a2 = torch.einsum("hsd,nhjd->nhsj", q, k2)  # q is stored pre-scaled
        pr = torch.exp((a2 - lse[None, ..., None]).masked_fill(~m, float("-inf")))
        X = torch.einsum("hsd,nhjd->nhsj", g, v2) - go[None, ..., None]
        pw = torch.exp(a[:, :, posc] - lse[..., None, None]).permute(2, 0, 1, 3) * m
        Y = (g @ v.transpose(-1, -2) - go[..., None])[:, :, posc].permute(2, 0, 1, 3)
        num = (pr * X - pw * Y).sum(-1)
        den = 1 + (pr - pw).sum(-1)
        total = total + ((num / den) * down[:, None, :]).sum(dim=(1, 2))
    return total


def occlusion(model, params, embeds, labels, token_loss, w, budget=1024, positions=None):
    """The approximate change in the summed loss when each position's embedding
    is zeroed, [len(positions)] (default every position), and the summed loss. embeds [1,T,d]; labels [T];
    token_loss(logits [N,L,V], targets [N,L]) -> [N,L] weighted per-token
    losses, 0 where the target is -100. The model must use
    attn_implementation="occlusion_probe"."""
    T = embeds.shape[1]
    dev = embeds.device
    nlayers = model.config.num_hidden_layers
    H = model.config.num_attention_heads
    D = getattr(model.config, "head_dim", None) or model.config.hidden_size // H

    def record(delta):
        PROBE.mode, PROBE.stash, PROBE.delta = "record", {}, delta
        logits = functional_call(model, params, (), {"inputs_embeds": embeds}).logits
        losses = token_loss(logits[:, :-1], labels[None, 1:])
        stash, PROBE.stash, PROBE.delta = PROBE.stash, None, None
        return losses.sum(), (losses[0], stash)

    zeros = [torch.zeros(1, H, T, D, dtype=embeds.dtype, device=dev) for _ in range(nlayers)]
    dO, (base, stash) = grad(record, has_aux=True)(zeros)
    total = base.sum()
    base = torch.cat([base, base.new_zeros(w + 1)])
    out = []
    positions = torch.arange(T, device=dev) if positions is None else positions
    for ts in positions.split(max(1, budget // w)):
        pos = ts[:, None] + torch.arange(w, device=dev)[None]
        posc = pos.clamp(max=T - 1)
        x = embeds[0][posc].clone()
        x[:, 0] = 0
        PROBE.mode, PROBE.stash, PROBE.ts, PROBE.window_kv = "window", stash, ts, {}
        logits = functional_call(model, params, (), {"inputs_embeds": x, "position_ids": posc}).logits
        kv, PROBE.window_kv, PROBE.stash, PROBE.mode = PROBE.window_kv, None, None, None
        targets = torch.where(pos + 1 < T, labels[(pos + 1).clamp(max=T - 1)], -100)
        win = token_loss(logits, targets)
        own = (win - base[pos] * (targets != -100)).sum(-1)
        out.append(own + _downstream(stash, dO, kv, ts, w))
    return torch.cat(out), total
