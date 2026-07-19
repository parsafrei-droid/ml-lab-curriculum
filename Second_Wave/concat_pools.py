import sys, torch, pathlib
BASE = pathlib.Path(__file__).parent
out = BASE / sys.argv[1]
parts = sys.argv[2:]
items, meta = [], None
for p in parts:
    d = torch.load(BASE / p, weights_only=False)
    items.extend(d["items"]); meta = meta or d["meta"]
meta = dict(meta); meta["size"] = len(items)
torch.save({"items": items, "meta": meta}, out)
feats = [it["n_features"] for it in items]
print(f"concatenated {len(items)} datasets -> {out} | features {min(feats)}..{max(feats)} mean {sum(feats)/len(feats):.1f}")
