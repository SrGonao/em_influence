import torch
from torch.func import functional_call, jvp
from transformers import Olmo3Config, Olmo3ForCausalLM

from em_influence import window_occlusion


def token_loss(logits, targets):
    flat = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.flatten(), reduction="none")
    return flat.reshape_as(targets)


def test_full_window_is_exact():
    torch.manual_seed(0)
    config = Olmo3Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, eos_token_id=None, pad_token_id=None,
                         attn_implementation="occlusion_probe")
    model = Olmo3ForCausalLM(config).double().eval()
    T = 20
    ids = torch.randint(0, 100, (T,))
    labels = ids.clone()
    labels[:8] = -100
    params = {k: p.detach() for k, p in model.named_parameters() if "proj" in k}
    direction = {k: torch.randn_like(p) for k, p in params.items()}
    embeds = model.get_input_embeddings()(ids)[None].detach()

    def score(e):
        def loss(p):
            window_occlusion.PROBE.mode, window_occlusion.PROBE.stash = "record", {}
            logits = functional_call(model, p, (), {"inputs_embeds": e}).logits
            return token_loss(logits[:, :-1], labels[None, 1:]).sum()

        return jvp(loss, (params,), (direction,))[1]

    with torch.no_grad():
        base = score(embeds)
        zeroed = []
        for t in range(T):
            e = embeds.clone()
            e[0, t] = 0
            zeroed.append(score(e) - base)
        _, (change, document) = jvp(
            lambda p: window_occlusion.occlusion(model, p, embeds, labels, token_loss, T, budget=64),
            (params,), (direction,))
        _, (chosen, _) = jvp(
            lambda p: window_occlusion.occlusion(model, p, embeds, labels, token_loss, 1, budget=64, extra=T),
            (params,), (direction,))
    torch.testing.assert_close(document, base, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(change, torch.stack(zeroed), rtol=1e-9, atol=1e-9)
    # Choosing every later position by attention is exact too.
    torch.testing.assert_close(chosen, torch.stack(zeroed), rtol=1e-9, atol=1e-9)

    with torch.no_grad():
        some = torch.tensor([0, 3, 7, 19])
        _, (subset, _) = jvp(
            lambda p: window_occlusion.occlusion(model, p, embeds, labels, token_loss, 4, budget=8, positions=some),
            (params,), (direction,))
        _, (every, _) = jvp(
            lambda p: window_occlusion.occlusion(model, p, embeds, labels, token_loss, 4, budget=64),
            (params,), (direction,))
    torch.testing.assert_close(subset, every[some], rtol=1e-10, atol=1e-10)
