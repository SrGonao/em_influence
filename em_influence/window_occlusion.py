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
from torch.func import functional_call, grad
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
        # One layer recomputed for its backward step, with the probe added to its output.
        o, _ = _CausalAttention.apply(query * scaling, k, v)
        return (o + PROBE.delta).transpose(1, 2), None
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


def _downstream(stash, dO, window_kv, pos, valid, in_set, per_query=False):
    """sum over layers, heads and queries s outside the set and after the zeroed
    token of dO_s . (change in s's attention output when the members' keys and
    values replace the originals)."""
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
        g = dO[i][0]
        go = (g * o).sum(-1)
        a2 = torch.einsum("hsd,nhjd->nhsj", q, k2)  # q is stored pre-scaled
        pr = torch.exp((a2 - lse[None, ..., None]).masked_fill(~m, float("-inf")))
        X = torch.einsum("hsd,nhjd->nhsj", g, v2) - go[None, ..., None]
        kw = k[:, posc]  # [H,n,m,D]
        pw = torch.exp(torch.einsum("hsd,hnjd->nhsj", q, kw) - lse[None, ..., None]) * m
        Y = torch.einsum("hsd,hnjd->nhsj", g, v[:, posc]) - go[None, ..., None]
        num = (pr * X - pw * Y).sum(-1)
        den = 1 + (pr - pw).sum(-1)
        total = total + ((num / den) * down[:, None, :]).sum(dim=1)
    return total if per_query else total.sum(-1)


def _members(stash, positions, T, w, extra, chosen=None):
    """Each candidate t's recomputed set: t..t+w-1, plus the `extra` later
    positions that attend to t most (attention summed over layers and heads),
    plus any `chosen` positions; duplicates and positions >= T are dropped."""
    dev = positions.device
    parts = [positions[:, None] + torch.arange(w, device=dev)[None]]
    if extra:
        mass = sum(c["mass"] for c in stash.values()).T[positions]  # [n, s]
        ar = torch.arange(T, device=dev)
        mass = mass.masked_fill(ar[None, :] < positions[:, None] + w, -1.0)
        values, picked = mass.topk(min(extra, T), dim=-1)
        parts.append(torch.where(values >= 0, picked, T))
    if chosen is not None:
        parts.append(torch.where(chosen >= positions[:, None] + w, chosen, T))
    pos = torch.cat(parts, -1).sort(-1).values
    repeated = torch.cat([torch.zeros_like(pos[:, :1], dtype=torch.bool), pos[:, 1:] == pos[:, :-1]], -1)
    pos = pos.masked_fill(repeated, T).sort(-1).values
    valid = pos < T
    in_set = torch.zeros(len(pos), T + 1, dtype=torch.bool, device=dev)
    in_set[torch.arange(len(pos), device=dev)[:, None], pos.clamp(max=T)] = True
    return pos, valid, in_set[:, :T]


def _record(model, params, embeds, labels, token_loss):
    """Each layer's keys, values and normalisers, the gradient of the summed
    loss at each layer's attention output, and the per-token losses. The
    backward goes one layer at a time from stored layer inputs, so only one
    layer's activations are held at once."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder = base.model
    T = embeds.shape[1]
    position_ids = torch.arange(T, device=embeds.device)[None]
    with _reparametrize_module(model, params):
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
        dO = {}
        PROBE.mode = "replay"
        for i in reversed(range(len(decoder.layers))):
            def replay(h, delta, g):
                PROBE.delta = delta
                return (layer(i, h) * g).sum()

            g, dO[i] = grad(replay, argnums=(0, 1))(inputs[i], torch.zeros_like(stash[i]["o"]), g)
        PROBE.mode, PROBE.stash, PROBE.delta = None, None, None
    return [dO[i] for i in range(len(dO))], losses, stash


def occlusion(model, params, embeds, labels, token_loss, w, budget=1024, positions=None, extra=0, parts=False,
              per_query=False, chosen=None):
    """The approximate change in the summed loss when each position's embedding
    is zeroed, [len(positions)] (default every position), and the summed loss.
    embeds [1,T,d]; labels [T]; token_loss(logits [N,L,V], targets [N,L]) ->
    [N,L] weighted per-token losses, 0 where the target is -100. The model must
    use attn_implementation="occlusion_probe". With parts, the change is
    [2, len(positions)]: the recomputed members' own loss change, and the tail.
    With per_query, it is the tail alone split by later query, [len(positions), T].
    chosen [len(positions), k] gives each token's extra members explicitly
    (positions >= T are ignored) instead of by attention."""
    T = embeds.shape[1]
    dev = embeds.device
    dO, base, stash = _record(model, params, embeds, labels, token_loss)
    total = base.sum()
    base = torch.cat([base, base.new_zeros(1)])
    out = []
    positions = torch.arange(T, device=dev) if positions is None else positions
    width = w + extra + (chosen.shape[1] if chosen is not None else 0)
    for start in range(0, len(positions), max(1, budget // width)):
        ts = positions[start:start + max(1, budget // width)]
        picked = None if chosen is None else chosen[start:start + len(ts)]
        pos, valid, in_set = _members(stash, ts, T, w, extra, picked)
        posc = pos.clamp(max=T - 1)
        x = embeds[0][posc].clone()
        x[:, 0] = 0
        PROBE.mode, PROBE.stash, PROBE.members, PROBE.window_kv = "window", stash, (pos, valid, in_set), {}
        logits = functional_call(model, params, (), {"inputs_embeds": x, "position_ids": posc}).logits
        kv, PROBE.window_kv, PROBE.stash, PROBE.mode = PROBE.window_kv, None, None, None
        targets = torch.where(valid & (pos + 1 < T), labels[(pos + 1).clamp(max=T - 1)], -100)
        win = token_loss(logits, targets)
        own = (win - base[posc] * (targets != -100)).sum(-1)
        tail = _downstream(stash, dO, kv, pos, valid, in_set, per_query)
        out.append(tail if per_query else torch.stack([own, tail]) if parts else own + tail)
    return torch.cat(out, 0 if per_query else -1), total


def sharpest_queries(per_query, positions, w, k):
    """[n, k]: for each token, the k later queries (at or after its window's end)
    whose tail terms are largest in magnitude, T where there are fewer."""
    T = per_query.shape[1]
    ar = torch.arange(T, device=per_query.device)
    size = per_query.abs().masked_fill(ar[None, :] < positions[:, None] + w, -1.0)
    values, index = size.topk(min(k, T), dim=-1)
    return torch.where(values > 0, index, T)
