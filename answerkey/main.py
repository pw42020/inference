"""Run one prompt through the model"""

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = Path("../models/smollm2-135m")
OUT = Path("../out")

if not os.path.exists(OUT):
    os.mkdir(OUT)

PROMPT = "What the dog doing?"

model = AutoModelForCausalLM.from_pretrained(MODEL)

tok = AutoTokenizer.from_pretrained(MODEL)

ids = tok(PROMPT, return_tensors="pt").input_ids

shapes = {}


def save(name, t):
    a = t.detach().float().squeeze(0).contiguous().numpy()
    a.tofile(OUT / f"{name}.bin")
    shapes[name] = list(a.shape)


# make hooks for submodules of the decoder layer
"""In PyTorch, a module (nn.Module) is a building block: an object that holds 
some weights and has a forward() function
computing its output. Modules can hold other modules as attributes, so a model is a
tree of them. A submodule is any module inside another one.

For SmolLM2 the tree looks like this:

LlamaForCausalLM            ← model
├── model: LlamaModel       ← model.model
│   ├── embed_tokens        (the embedding table)
│   ├── layers              (a list of 30)
│   │   └── [0] LlamaDecoderLayer      ← l0
│   │       ├── input_layernorm        (RMSNorm)
│   │       ├── self_attn
│   │       │   ├── q_proj  (Linear: a matrix multiply)
│   │       │   ├── k_proj
│   │       │   ├── v_proj
│   │       │   └── o_proj
│   │       ├── post_attention_layernorm
│   │       └── mlp
│   │           ├── gate_proj
│   │           ├── up_proj
│   │           └── down_proj
│   └── norm                (the final RMSNorm)
└── lm_head                 (the output matrix, shared with embed_tokens here)
"""

l0 = model.model.layers[0]
hooks = []

submodules = {
    "l0_input_norm": l0.input_layernorm,
    "l0_q": l0.self_attn.q_proj,  # before RoPE
    "l0_k": l0.self_attn.k_proj,  # before RoPE
    "l0_v": l0.self_attn.v_proj,
    "l0_attn": l0.self_attn,  # after the o projection, before the residual add
    "l0_post_norm": l0.post_attention_layernorm,
    "l0_mlp": l0.mlp,
}.items()

for name, mod in submodules:
    hooks.append(
        mod.register_forward_hook(
            lambda m, i, o, name=name: save(name, o[0] if isinstance(o, tuple) else o)
        )
    )

with torch.no_grad():
    out = model(ids, output_hidden_states=True)

for h in hooks:
    h.remove()

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
