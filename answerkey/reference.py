# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers", "numpy"]
# ///
"""Run one prompt through the model and save every step as raw little-endian f32."""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = Path(sys.argv[1] if len(sys.argv) > 1 else "models/smollm2-135m")
OUT = Path("ref")
PROMPT = "The robot drove into the kitchen and"

OUT.mkdir(exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
ids = tok(PROMPT, return_tensors="pt").input_ids
shapes = {}


def save(name, t):
    a = t.detach().float().squeeze(0).contiguous().numpy()
    a.tofile(OUT / f"{name}.bin")
    shapes[name] = list(a.shape)


# Steps inside layer 0, for when a layer doesn't match and you need to know where.
l0 = model.model.layers[0]
hooks = [
    mod.register_forward_hook(
        lambda m, i, o, name=name: save(name, o[0] if isinstance(o, tuple) else o)
    )
    for name, mod in {
        "l0_input_norm": l0.input_layernorm,
        "l0_q": l0.self_attn.q_proj,  # before RoPE
        "l0_k": l0.self_attn.k_proj,  # before RoPE
        "l0_v": l0.self_attn.v_proj,
        "l0_attn": l0.self_attn,  # after the o projection, before the residual add
        "l0_post_norm": l0.post_attention_layernorm,
        "l0_mlp": l0.mlp,
    }.items()
]

with torch.no_grad():
    out = model(ids, output_hidden_states=True)
for h in hooks:
    h.remove()  # or generate() below overwrites them

hs = out.hidden_states  # [embeddings, after layer 0, ..., norm(after last layer)]
save("embed", hs[0])
for i in range(1, len(hs) - 1):
    save(f"layer{i - 1:02d}", hs[i])
save("final_norm", hs[-1])
save("logits", out.logits)

with torch.no_grad():
    gen = model.generate(ids, max_new_tokens=20, do_sample=False)

(OUT / "ids.json").write_text(json.dumps(ids[0].tolist()))
(OUT / "greedy.json").write_text(json.dumps(gen[0].tolist()))
(OUT / "shapes.json").write_text(json.dumps(shapes, indent=1))
print(ids[0].tolist())
print(tok.decode(gen[0]))
