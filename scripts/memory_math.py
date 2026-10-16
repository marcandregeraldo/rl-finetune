"""Memory math for the rl-finetune target models: params, KV cache, fine-tuning, RL sizing.

Configs copied from the Hugging Face config.json files (text_config only; the ~0.4B Pixtral
vision tower is ignored unless noted). Run: python3 scripts/memory_math.py

Every number is an estimate to reason with, not a guarantee: frameworks add buffers,
fragmentation and CUDA context (~1-3 GB per GPU). Verify by measuring (PLAN.md M1.5).
Study sheet with formulas and exercises: resume repo, study/memory-math.md.
"""

GB = 1e9

MODELS = {
    # Dense, GQA. Source: mistralai/Ministral-3-*-Instruct-2512-BF16/config.json
    "Ministral-3-3B": dict(kind="dense", L=26, h=3072, I=9216, nh=32, nkv=8, hd=128, V=131072, tied=True),
    "Ministral-3-8B": dict(kind="dense", L=34, h=4096, I=14336, nh=32, nkv=8, hd=128, V=131072, tied=False),
    "Ministral-3-14B": dict(kind="dense", L=40, h=5120, I=16384, nh=32, nkv=8, hd=128, V=131072, tied=False),
    # MoE + MLA. Source: mistralai/Mistral-Small-4-119B-2603/config.json
    "Mistral-Small-4-119B": dict(kind="moe_mla", L=36, h=4096, nh=32, V=131072, tied=False,
                                 q_lora=1024, kv_lora=256, qk_nope=64, qk_rope=64, v_hd=128,
                                 n_exp=128, n_shared=1, top_k=4, moe_I=2048),
}

GPUS = {"H100-80GB": 80, "H200-141GB": 141, "B200-192GB": 192, "A100-80GB": 80, "L40S-48GB": 48,
        "M4 Pro 24GB (Mac, ~17GB usable)": 17}


def params(m):
    """Returns (total, active, embedding) parameter counts for the text model."""
    emb = m["V"] * m["h"] * (1 if m["tied"] else 2)
    h, L = m["h"], m["L"]
    if m["kind"] == "dense":
        attn = h * m["nh"] * m["hd"] + 2 * h * m["nkv"] * m["hd"] + m["nh"] * m["hd"] * h
        mlp = 3 * h * m["I"]
        per_layer = attn + mlp
        return L * per_layer + emb, L * per_layer + emb, emb
    # MLA attention (DeepSeek-style): low-rank q, joint low-rank kv, decoupled rope key
    qk_hd = m["qk_nope"] + m["qk_rope"]
    attn = (h * m["q_lora"] + m["q_lora"] * m["nh"] * qk_hd            # q_a, q_b
            + h * (m["kv_lora"] + m["qk_rope"])                         # kv_a (+ shared rope key)
            + m["kv_lora"] * m["nh"] * (m["qk_nope"] + m["v_hd"])       # kv_b
            + m["nh"] * m["v_hd"] * h)                                  # o
    expert = 3 * h * m["moe_I"]
    router = h * m["n_exp"]
    total_layer = attn + (m["n_exp"] + m["n_shared"]) * expert + router
    active_layer = attn + (m["top_k"] + m["n_shared"]) * expert + router
    return L * total_layer + emb, L * active_layer + emb, emb


def kv_bytes_per_token(m, dtype_bytes=2):
    """KV cache per token, all layers."""
    if m["kind"] == "dense":
        return 2 * m["L"] * m["nkv"] * m["hd"] * dtype_bytes          # K and V, per KV head
    # MLA caches the compressed latent + the shared rope key, not per-head K/V
    return m["L"] * (m["kv_lora"] + m["qk_rope"]) * dtype_bytes


def kv_if_mha(m, dtype_bytes=2):
    """What the cache would be with plain multi-head attention (no GQA/MLA), for comparison."""
    hd = m.get("hd", m.get("v_hd"))
    return 2 * m["L"] * m["nh"] * hd * dtype_bytes


def finetune_gb(n_params, method, trainable_frac=0.0):
    """Static training memory (weights + grads + optimizer), no activations.

    full: bf16 weights 2 + bf16 grads 2 + fp32 master 4 + Adam m,v 8 = 16 B/param
          (verl/FSDP keeps fp32 params + fp32 grads: 4 + 4 + 8 = 16 B/param, same total)
    lora: frozen bf16 base 2 B/param; trainable LoRA params at 16 B/param
    qlora: frozen 4-bit NF4 base ~0.55 B/param (incl. block scales); LoRA at 16 B/param
    """
    base = {"full": 16, "lora": 2, "qlora": 0.55}[method]
    if method == "full":
        return n_params * 16 / GB
    return (n_params * base + n_params * trainable_frac * 16) / GB


def lora_params(m, r, targets=("q", "k", "v", "o", "gate", "up", "down")):
    """LoRA trainable params for a dense model: r * (d_in + d_out) per adapted matrix."""
    h, L = m["h"], m["L"]
    shapes = {"q": (h, m["nh"] * m["hd"]), "k": (h, m["nkv"] * m["hd"]), "v": (h, m["nkv"] * m["hd"]),
              "o": (m["nh"] * m["hd"], h), "gate": (h, m["I"]), "up": (h, m["I"]), "down": (m["I"], h)}
    return L * sum(r * (a + b) for a, b in (shapes[t] for t in targets))


def activations_gb(m, tokens, checkpointing=True, dtype_bytes=2):
    """Rough activation memory for a training microbatch of `tokens` total tokens.

    With full activation checkpointing: keep each layer's input (tokens * h) plus one layer's
    working set recomputed in backward (~34 * tokens * h bytes, Korthikanti et al. without
    attention scores; flash attention avoids the s^2 term).
    Without: ~34 * tokens * h bytes per layer.
    """
    h, L = m["h"], m["L"]
    per_layer_full = 34 * tokens * h
    if checkpointing:
        return (L * tokens * h * dtype_bytes + per_layer_full) / GB
    return L * per_layer_full / GB


def logits_gb(m, tokens, dtype_bytes=4):
    """Full-vocab logits in fp32: the RL logprob gotcha (131k vocab)."""
    return tokens * m["V"] * dtype_bytes / GB


def fmt(x):
    return f"{x:,.1f}"


def main():
    print("## Parameters (text model)\n")
    print("| Model | Total | Active / token | Embeddings | bf16 weights | fp8 weights |")
    print("|---|---|---|---|---|---|")
    P = {}
    for name, m in MODELS.items():
        tot, act, emb = params(m)
        P[name] = (tot, act)
        print(f"| {name} | {tot/1e9:.2f}B | {act/1e9:.2f}B | {emb/1e9:.2f}B | {fmt(tot*2/GB)} GB | {fmt(tot/GB)} GB |")

    print("\n## KV cache\n")
    print("| Model | Attention | KV / token (bf16) | 32k-token seq | 1M tokens in flight | Same model with MHA |")
    print("|---|---|---|---|---|---|")
    for name, m in MODELS.items():
        b = kv_bytes_per_token(m)
        att = "GQA 32q/8kv" if m["kind"] == "dense" else "MLA (latent 256 + rope 64)"
        print(f"| {name} | {att} | {b/1024:.1f} KB | {fmt(b*32768/GB)} GB | {fmt(b*1e6/GB)} GB | {kv_if_mha(m)/1024:.0f} KB/token |")

    print("\n## Serving: how many tokens of KV fit (90% util, 2 GB overhead; weight dtype per row)\n")
    print("| Model | GPU setup | Weights / GPU | KV budget / GPU | Tokens of KV (whole setup) | Concurrent 8k seqs |")
    print("|---|---|---|---|---|---|")
    serving = [("Ministral-3-3B", "H100-80GB", 1, 2), ("Ministral-3-8B", "L40S-48GB", 1, 2),
               ("Ministral-3-8B", "H100-80GB", 1, 2), ("Ministral-3-14B", "H100-80GB", 1, 2),
               ("Mistral-Small-4-119B", "H100-80GB", 2, 1), ("Mistral-Small-4-119B", "H100-80GB", 4, 2),
               ("Mistral-Small-4-119B", "B200-192GB", 1, 1)]
    for name, gpu, tp, wbytes in serving:
        m = MODELS[name]
        tot, _ = P[name]
        w = tot * wbytes / GB / tp
        budget = GPUS[gpu] * 0.9 - 2 - w
        if budget <= 0:
            print(f"| {name} | {tp}× {gpu}, {'bf16' if wbytes==2 else 'fp8'} | {fmt(w)} GB | doesn't fit | — | — |")
            continue
        # KV shards across TP for dense GQA (heads split); MLA latent is replicated per TP rank
        kv_tok = kv_bytes_per_token(m) / (tp if m["kind"] == "dense" else 1)
        tokens = budget * GB / kv_tok
        print(f"| {name} | {tp}× {gpu}, {'bf16' if wbytes==2 else 'fp8'} | {fmt(w)} GB | {fmt(budget)} GB | {tokens/1e6:.2f}M | {int(tokens // 8192):,} |")

    print("\n## Fine-tuning: static memory (weights + grads + Adam), before activations\n")
    print("| Model | Full FT (16 B/param) | LoRA r=16, all linear | QLoRA r=16, all linear | LoRA trainable params |")
    print("|---|---|---|---|---|")
    for name, m in MODELS.items():
        tot, _ = P[name]
        if m["kind"] == "dense":
            lp = lora_params(m, 16)
            frac = lp / tot
            print(f"| {name} | {fmt(finetune_gb(tot,'full'))} GB | {fmt(finetune_gb(tot,'lora',frac))} GB | {fmt(finetune_gb(tot,'qlora',frac))} GB | {lp/1e6:.1f}M |")
        else:
            print(f"| {name} | {fmt(finetune_gb(tot,'full'))} GB (~24 H100s just for states) | {fmt(tot*2/GB)} GB base + adapters | {fmt(tot*0.55/GB)} GB base + adapters | (attention-only LoRA is the usual MoE choice) |")

    print("\n## Activations and logits per microbatch (gradient checkpointing on)\n")
    print("| Model | 4 × 2,048 tokens: activations | same: fp32 logits if materialised | 1 × 8,192: activations |")
    print("|---|---|---|---|")
    for name, m in MODELS.items():
        if m["kind"] != "dense":
            continue
        print(f"| {name} | {fmt(activations_gb(m, 8192))} GB | {fmt(logits_gb(m, 8192))} GB | {fmt(activations_gb(m, 8192))} GB |")

    print("\n## GRPO step sizing on H100-80GB (verl-style colocated, two peaks)\n")
    print("Batch: 64 prompts × group 8 = 512 sequences of 512 prompt + 1,536 response = 2,048 tokens.")
    print("Training peak: sharded actor states + sharded ref weights + activations and fp32 logits for a")
    print("4 × 2,048 microbatch (logits chunked in practice; shown unchunked). Rollout peak: engine")
    print("weights (TP-sharded) + KV, with the actor offloaded (verl sleep / param+optimizer offload);")
    print("the engine gets ~60% of each GPU. Seqs that don't fit run in waves.\n")
    print("| Model | Method | GPUs (TP) | Training peak / GPU | Rollout: KV budget / replica | Concurrent seqs / replica | Waves for 512 | Verdict |")
    print("|---|---|---|---|---|---|---|---|")
    seqs, seq_len = 512, 2048
    plans = [("Ministral-3-3B", "full", 2, 1), ("Ministral-3-8B", "full", 2, 1), ("Ministral-3-8B", "full", 4, 2),
             ("Ministral-3-8B", "lora", 1, 1), ("Ministral-3-14B", "full", 4, 2), ("Ministral-3-14B", "full", 8, 2),
             ("Ministral-3-14B", "lora", 2, 2)]
    for name, method, ngpu, tp in plans:
        m = MODELS[name]
        tot, _ = P[name]
        if method == "full":
            states = finetune_gb(tot, "full") / ngpu
            ref = tot * 2 / GB / ngpu
        else:
            states = finetune_gb(tot, "lora", lora_params(m, 16) / tot) / ngpu  # FSDP shards the frozen base too
            ref = 0.0  # reference = adapter disabled
        train_peak = states + ref + activations_gb(m, 4 * seq_len) + logits_gb(m, 4 * seq_len) + 2
        replicas = ngpu // tp
        kv_budget = (80 * 0.6 - tot * 2 / GB / tp) * tp
        conc = int(kv_budget * GB // (kv_bytes_per_token(m) * seq_len)) if kv_budget > 0 else 0
        waves = -(-seqs // max(1, conc * replicas))
        verdict = "fits" if train_peak < 70 else ("tight: offload optimizer or add GPUs" if train_peak < 85 else "doesn't fit")
        print(f"| {name} | {method} | {ngpu}× ({tp}) | {fmt(train_peak)} GB | {fmt(kv_budget)} GB | {conc} | {waves} | {verdict} |")


if __name__ == "__main__":
    main()
