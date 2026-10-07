"""Windowed occlusion for any HF decoder that routes attention through
ALL_ATTENTION_FUNCTIONS (Llama, Qwen, OLMo-2/3, ...).

For every position t, approximates L(x with t's embedding zeroed) - L(x):
positions t..t+w-1, and optionally the positions that attend to t most, are
recomputed exactly (they attend to the original keys and values outside that
set; nothing before t changes when t is zeroed). The other later positions see
the set's new keys/values, and the resulting change in each layer's attention
output is computed in closed form and taken to the loss with the gradient at
that output. w >= T is exact.
"""

from __future__ import annotations

import inspect

import torch
from torch.func import grad
from torch.nn.utils.stateless import _reparametrize_module
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class _CausalAttention(torch.autograd.Function):
    """softmax(q k^T, causal) v that saves q, k, v and each query's
    log-normaliser instead of the T x T probabilities, and recomputes them in
    the backward and forward-mode rules. q is pre-scaled."""

    @staticmethod
    def forward(q, k, v):
        a = _causal_logits(q, k)
        lse = torch.logsumexp(a, -1)
        return torch.exp(a - lse[..., None]) @ v, lse

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs, *output)
        ctx.save_for_forward(*inputs, *output)

    @staticmethod
    def backward(ctx, grad_o, grad_lse):
        q, k, v, o, lse = ctx.saved_tensors
        p = torch.exp(_causal_logits(q, k) - lse[..., None])
        ds = p * (grad_o @ v.transpose(-1, -2) - (grad_o * o).sum(-1, keepdim=True)) + p * grad_lse[..., None]
        return ds @ k, ds.transpose(-1, -2) @ q, p.transpose(-1, -2) @ grad_o

    @staticmethod
    def jvp(ctx, dq, dk, dv):
        q, k, v, o, lse = ctx.saved_tensors
        p = torch.exp(_causal_logits(q, k) - lse[..., None])
        da = torch.nan_to_num(_causal_logits(dq, k) + _causal_logits(q, dk), neginf=0.0)
        dlse = (p * da).sum(-1)
        return p * (da - dlse[..., None]) @ v + p @ dv, dlse


def _causal_logits(q, k):
    T = q.shape[-2]
    causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    return (q @ k.transpose(-1, -2)).masked_fill(~causal, float("-inf"))


class _Probe:
    mode = None
    stash = None
    delta = None
    members = None  # (positions [n,m] sorted, first is the zeroed token; valid [n,m]; in_set [n,T])
    window_kv = None


PROBE = _Probe()


def probe_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, sliding_window=None, **kwargs):
    i = module.layer_idx
    k = key.repeat_interleave(module.num_key_value_groups, 1)
    v = value.repeat_interleave(module.num_key_value_groups, 1)
    if PROBE.mode == "replay":
        o, _ = _CausalAttention.apply(query * scaling, k, v)
        return o.transpose(1, 2), None
    if PROBE.mode == "record":
        T = query.shape[2]
        if sliding_window is not None and T > sliding_window:
            raise ValueError("documents longer than the sliding window aren't supported")
        q = query * scaling
        o, lse = _CausalAttention.apply(q, k, v)
        with torch.no_grad():
            mass = torch.exp(_causal_logits(q, k) - lse[..., None]).sum(1)[0]
        # Only what the tail needs, not the T x T logits: each query's log-normaliser,
        # and the attention mass each key gets (for choosing extra members).
        PROBE.stash[i] = dict(q=q, k=k, v=v, lse=lse, o=o, mass=mass)
        return o.transpose(1, 2), None
    # Each copy's recomputed members attend to the original keys outside the set
    # and causally to each other.
    c = PROBE.stash[i]
    pos, valid, in_set = PROBE.members
    m = pos.shape[1]
    T = c["k"].shape[2]
    dev = query.device
    before = torch.arange(T, device=dev)[None, None, :] < pos.clamp(max=T - 1)[:, :, None]
    outside = before & ~in_set[:, None, :]  # [n,m,T]
    causal = torch.ones(m, m, dtype=torch.bool, device=dev).tril()[None] & valid[:, None, :]
    a1 = torch.einsum("nhjd,hsd->nhjs", query * scaling, c["k"][0])
    a1 = a1.masked_fill(~outside[:, None], float("-inf"))
    a2 = (query @ k.transpose(-1, -2) * scaling).masked_fill(~causal[:, None], float("-inf"))
    p = torch.cat([a1, a2], -1).softmax(-1)
    o = torch.einsum("nhjs,hsd->nhjd", p[..., :T], c["v"][0]) + p[..., T:] @ v
    PROBE.window_kv[i] = (k, v)
    return o.transpose(1, 2), None


ALL_ATTENTION_FUNCTIONS.register("occlusion_probe", probe_attention)


def _branch(layer):
    """The module whose output the attention block adds to the residual stream:
    the post-attention norm in post-norm models (OLMo-2/3), else o_proj."""
    return layer.post_attention_layernorm if hasattr(layer, "post_feedforward_layernorm") else layer.self_attn.o_proj


def _attention_to_residual(layer, o):
    """o [..., H, S, D] -> what the attention block adds to the residual, [..., S, d]."""
    x = o.transpose(-3, -2).flatten(-2)
    x = layer.self_attn.o_proj(x)
    return layer.post_attention_layernorm(x) if hasattr(layer, "post_feedforward_layernorm") else x


def _downstream(layers, stash, dR, window_kv, pos, valid, in_set):
    """sum over layers and queries s outside the set and after the zeroed token
    of dR_s . (change in what s's attention block adds to the residual stream
    when the members' keys and values replace the originals). The attention
    output is renormalised exactly and taken through o_proj and any
    post-attention norm exactly; only what follows is linearised."""
    T = stash[0]["k"].shape[2]
    dev = pos.device
    posc = pos.clamp(max=T - 1)
    ar = torch.arange(T, device=dev)
    down = (ar[None, :] > pos[:, :1]) & ~in_set  # [n,s]
    m = (valid[:, None, :] & (posc[:, None, :] < ar[None, :, None]) & down[:, :, None])[:, None]  # [n,1,s,j]
    total = 0
    for i in sorted(stash):
        c = stash[i]
        k2, v2 = window_kv[i]  # [n,H,m,D]
        q, k, o, v, lse = c["q"][0], c["k"][0], c["o"][0], c["v"][0], c["lse"][0]
        pr = torch.exp((torch.einsum("hsd,nhjd->nhsj", q, k2) - lse[None, ..., None]).masked_fill(~m, float("-inf")))
        pw = torch.exp(torch.einsum("hsd,hnjd->nhsj", q, k[:, posc]) - lse[None, ..., None]) * m
        den = 1 + (pr - pw).sum(-1)
        new = (o[None] - torch.einsum("nhsj,hnjd->nhsd", pw, v[:, posc]) + torch.einsum("nhsj,nhjd->nhsd", pr, v2))
        new = new / den[..., None]
        change = _attention_to_residual(layers[i], new) - _attention_to_residual(layers[i], o[None])
        total = total + (change * dR[i][0] * down[..., None]).sum(dim=(1, 2))
    return total


def _members(stash, positions, T, w, extra):
    """Each candidate t's recomputed set: t..t+w-1, plus the `extra` later
    positions that attend to t most (attention summed over layers and heads)."""
    dev = positions.device
    pos = positions[:, None] + torch.arange(w, device=dev)[None]
    if extra:
        mass = sum(c["mass"] for c in stash.values()).T[positions]  # [n, s]
        ar = torch.arange(T, device=dev)
        mass = mass.masked_fill(ar[None, :] < positions[:, None] + w, -1.0)
        values, chosen = mass.topk(min(extra, T), dim=-1)
        pos = torch.cat([pos, torch.where(values >= 0, chosen, T)], -1).sort(-1).values
    valid = pos < T
    in_set = torch.zeros(len(pos), T + 1, dtype=torch.bool, device=dev)
    in_set[torch.arange(len(pos), device=dev)[:, None], pos.clamp(max=T)] = True
    return pos, valid, in_set[:, :T]


def _record(model, embeds, labels, token_loss):
    """Each layer's keys, values and normalisers, the gradient of the summed
    loss at what each layer's attention block adds to the residual stream, the
    per-token losses, and the decoder layers. The
    backward goes one layer at a time from stored layer inputs, so only one
    layer's activations are held at once."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder = base.model
    T = embeds.shape[1]
    position_ids = torch.arange(T, device=embeds.device)[None]
    if True:
        layer_types = getattr(base.config, "layer_types", None)
        if "layer_type" in inspect.signature(decoder.rotary_emb.forward).parameters:
            rope = {kind: decoder.rotary_emb(embeds, position_ids, kind) for kind in set(layer_types)}
            position_embeddings = [rope[kind] for kind in layer_types]
        else:
            position_embeddings = [decoder.rotary_emb(embeds, position_ids)] * len(decoder.layers)

        def layer(i, h):
            out = decoder.layers[i](h, attention_mask=None, position_ids=position_ids,
                                    position_embeddings=position_embeddings[i])
            return out[0] if isinstance(out, tuple) else out

        def head(h):
            losses = token_loss(base.lm_head(decoder.norm(h))[:, :-1], labels[None, 1:])
            return losses.sum(), losses[0]

        PROBE.mode, PROBE.stash = "record", {}
        inputs = [embeds]
        for i in range(len(decoder.layers)):
            inputs.append(layer(i, inputs[-1]))
        stash = PROBE.stash
        g, losses = grad(head, has_aux=True)(inputs[-1])
        dR = {}
        PROBE.mode = "replay"
        for i in reversed(range(len(decoder.layers))):
            def replay(h, delta, g):
                hook = _branch(decoder.layers[i]).register_forward_hook(lambda module, args, out: out + delta)
                try:
                    return (layer(i, h) * g).sum()
                finally:
                    hook.remove()

            g, dR[i] = grad(replay, argnums=(0, 1))(inputs[i], torch.zeros_like(inputs[i]), g)
        PROBE.mode, PROBE.stash = None, None
    return [dR[i] for i in range(len(dR))], losses, stash, list(decoder.layers)


def occlusion(model, params, embeds, labels, token_loss, w, budget=1024, positions=None, extra=0):
    """The approximate change in the summed loss when each position's embedding
    is zeroed, [len(positions)] (default every position), and the summed loss.
    embeds [1,T,d]; labels [T]; token_loss(logits [N,L,V], targets [N,L]) ->
    [N,L] weighted per-token losses, 0 where the target is -100. The model must
    use attn_implementation="occlusion_probe"."""
    T = embeds.shape[1]
    dev = embeds.device
    with _reparametrize_module(model, params):
        return _occlusion(model, embeds, labels, token_loss, w, budget, positions, extra)


def _occlusion(model, embeds, labels, token_loss, w, budget, positions, extra):
    T = embeds.shape[1]
    dev = embeds.device
    dR, base, stash, layers = _record(model, embeds, labels, token_loss)
    total = base.sum()
    base = torch.cat([base, base.new_zeros(1)])
    out = []
    positions = torch.arange(T, device=dev) if positions is None else positions
    for ts in positions.split(max(1, budget // (w + extra))):
        pos, valid, in_set = _members(stash, ts, T, w, extra)
        posc = pos.clamp(max=T - 1)
        x = embeds[0][posc].clone()
        x[:, 0] = 0
        PROBE.mode, PROBE.stash, PROBE.members, PROBE.window_kv = "window", stash, (pos, valid, in_set), {}
        logits = model(inputs_embeds=x, position_ids=posc).logits
        kv, PROBE.window_kv, PROBE.stash, PROBE.mode = PROBE.window_kv, None, None, None
        targets = torch.where(valid & (pos + 1 < T), labels[(pos + 1).clamp(max=T - 1)], -100)
        win = token_loss(logits, targets)
        own = (win - base[posc] * (targets != -100)).sum(-1)
        out.append(own + _downstream(layers, stash, dR, kv, pos, valid, in_set))
    return torch.cat(out), total
