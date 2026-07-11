#!/usr/bin/env python
"""Collect expert-routing traces from OLMoE-1B-7B (16 layers, 64 experts, top-8).

For every forward step, hooks on each layer's router (mlp.gate Linear) capture:
  - router logits (float16)             -> routing decisions, margins
  - the gate's input hidden state (fp16) -> cross-layer predictability probes

Output per prompt under traces/olmoe/<domain>/<id>/:
  decode_logits.npy  (T, L, E) fp16     decode steps
  decode_hidden.npy  (T, L, H) fp16
  prefill_logits.npy (P, L, E) fp16     prompt tokens (locality of prefill)
  meta.json

Usage: trace_olmoe.py --domains code,math --max-new 1024 [--threads 8]
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models/olmoe-1b-7b-instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="all")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="max prompts per domain (0=all)")
    ap.add_argument("--out", default=str(ROOT / "traces/olmoe"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    model.eval()

    layers = model.model.layers
    L = len(layers)
    E = model.config.num_experts
    H = model.config.hidden_size
    print(f"layers={L} experts={E} hidden={H} top_k={model.config.num_experts_per_tok}")

    # step_buf[layer] = (logits fp16 ndarray (T,E), hidden fp16 ndarray (T,H))
    step_buf = {}

    def mk_hook(layer_idx):
        def hook(_mod, inputs, output):
            hid = inputs[0].detach().to(torch.float16).cpu().numpy()
            # transformers 5.x OlmoeTopKRouter returns (router_logits, topk_w, topk_i)
            log = (output[0] if isinstance(output, tuple) else output)
            log = log.detach().to(torch.float16).cpu().numpy()
            step_buf[layer_idx] = (log, hid)
        return hook

    handles = [ly.mlp.gate.register_forward_hook(mk_hook(i)) for i, ly in enumerate(layers)]

    prompts = json.loads((ROOT / "scripts/prompts.json").read_text())
    domains = list(prompts) if args.domains == "all" else args.domains.split(",")

    for domain in domains:
        plist = prompts[domain][: args.limit] if args.limit else prompts[domain]
        for pid, prompt in enumerate(plist):
            outdir = Path(args.out) / domain / f"{pid:02d}"
            if (outdir / "meta.json").exists():
                print(f"skip {domain}/{pid:02d} (done)")
                continue
            outdir.mkdir(parents=True, exist_ok=True)

            msgs = [{"role": "user", "content": prompt}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            if not torch.is_tensor(ids):
                ids = ids["input_ids"]
            P = ids.shape[1]

            prefill_logits = None
            dec_logits, dec_hidden = [], []
            t0 = time.time()

            def collect(is_prefill):
                nonlocal prefill_logits
                logs = np.stack([step_buf[i][0] for i in range(L)], axis=1)  # (T,L,E)
                hids = np.stack([step_buf[i][1] for i in range(L)], axis=1)  # (T,L,H)
                if is_prefill:
                    prefill_logits = logs
                else:
                    dec_logits.append(logs)
                    dec_hidden.append(hids)
                step_buf.clear()

            with torch.no_grad():
                past = None
                cur = ids
                for step in range(args.max_new + 1):
                    out = model(cur, past_key_values=past, use_cache=True)
                    collect(is_prefill=(step == 0))
                    past = out.past_key_values
                    logits = out.logits[:, -1, :].float()
                    # temperature 0.7 sampling for realistic diversity
                    probs = torch.softmax(logits / 0.7, dim=-1)
                    nxt = torch.multinomial(probs, 1)
                    if nxt.item() == tok.eos_token_id:
                        break
                    cur = nxt
                    ids = torch.cat([ids, nxt], dim=1)

            T = len(dec_logits)
            np.save(outdir / "decode_logits.npy", np.concatenate(dec_logits, 0) if T else np.zeros((0, L, E), np.float16))
            np.save(outdir / "decode_hidden.npy", np.concatenate(dec_hidden, 0) if T else np.zeros((0, L, H), np.float16))
            np.save(outdir / "prefill_logits.npy", prefill_logits)
            text = tok.decode(ids[0, P:], skip_special_tokens=True)
            (outdir / "meta.json").write_text(json.dumps({
                "domain": domain, "pid": pid, "prompt": prompt,
                "prefill_tokens": int(P), "decode_tokens": T,
                "tok_per_s": round(T / max(time.time() - t0, 1e-9), 2),
                "output_preview": text[:400],
            }, indent=1))
            print(f"{domain}/{pid:02d}: P={P} T={T} {T/max(time.time()-t0,1e-9):.1f} tok/s")

    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
