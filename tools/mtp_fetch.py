"""tools/mtp_fetch.py - `verify` only: the MTP block's tensors against their SHA256, offline.

The GSQ-RCO GGUF ships no MTP head. The BF16 checkpoint (Qwen/Qwen3.8-Flash-Next) does: 31 `mtp.*` tensors. This tool
does not download anything. The tensors are one raw file per tensor in DIR/tensors (the layout an earlier fetch wrote,
with mtp-manifest.json beside it); `verify` checks them against the pinned revision's hashes:

    python tools/mtp_fetch.py verify --out DIR             # the fetched tensors against SHA256 (exit 3: bad ones)

#327: a range read that a mirror or proxy answered wrongly left an install with the right size and wrong bytes, so
every tensor of the pinned revision is checked against SHA256 below, and a corrupt one is reported here.
"""
import argparse
import hashlib
import json
import os
import sys

# #214: the checkpoint revision the hashes below are of (its `sha` from https://huggingface.co/api/models/Qwen/
# Qwen3.8-Flash-Next on 2026-09-30). STRATA_MTP_REVISION names another revision: verify then has nothing to check.
PINNED_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
REVISION = os.environ.get("STRATA_MTP_REVISION") or PINNED_REVISION
# STRATA_SOURCE=modelscope (setup's --source) serves the repository's current files, not a pinned revision: they are
# checked against the pinned revision's SHA256 all the same (pinned() is True for it).
MODELSCOPE = (os.environ.get("STRATA_SOURCE") or "").strip().lower() in ("ms", "modelscope")

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "I64": 8, "I32": 4}
BAD = 3                                             # `verify`'s exit code: a tensor is missing or corrupt

# #327: sha256 of every MTP tensor's bytes at PINNED_REVISION (two independent fetches agree, and 29 of them were read
# again from that revision; the two expert tensors' first and last 16 MiB), so a mirror that ignored the
# Range header (it sent the shard's start: its JSON header) is caught, and an install it corrupted is reported by verify.
SHA256 = {
    "mtp.fc_embedding.weight":
        "241d6e87ded317a61b1aeb3ca4e28c53f1caba09ad5be66092e91fe5e3e14751",
    "mtp.fc_hidden.weight":
        "97c8e427ad0177e2c713ed24a397d815b523eda15ef453f1ae6463f6c70a8f96",
    "mtp.hyper_connection_mixer.hc_norm.weight":
        "ce948a398e15aeaa403c424cda34e6eed296ab2570d591bb5ff3da642b255d64",
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight":
        "c633d2841fda463e82045e0b93fb8cb8d511ec0c9eda616020f01aca18d71947",
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight":
        "6565285f0ba2923703252763cfc11d8a9bda40ab42e7548e12464b04dbdbec13",
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight":
        "c8afff64e7f54412d02120097c4255033c55245e1c1eed84cc630d428ccb864c",
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight":
        "1fb619aeaaf637ccd70f46f1f8f8116e6a46448cb43026b246de8ee8286ed809",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight":
        "e80ac8bf3e270e9d5a949ec7c913329dc8de017d69851334820fae7b8c86a7f5",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight":
        "f80f1fd44e1183e7764ea496b11ff89ead9dd758f861d4e79a770bf54956b1a5",
    "mtp.layers.0.mlp.experts.down_proj":
        "539c46234f73bac70450f731f40e715f77ac32faf77def0f3f800e58aaabbbc0",
    "mtp.layers.0.mlp.experts.gate_up_proj":
        "839e98f9642ebc884b178abd2090b6dc8b0bcd81ed92d7f7a08314c6a68f96d5",
    "mtp.layers.0.mlp.gate.weight":
        "f0d35fa41b360a08bc0c5b1e5766aa42df960291716ee189208b5d2599462c20",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight":
        "d2ced8c81367cc494599d1a8cdc279c3dd6579b628ded05acd7495e99cee01d8",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight":
        "f4a81109165853b3b64f36d809f98094b599763364070f5fa079200e1c386d4c",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight":
        "172586861eea53efbff10afc43ded5b6e49c83879e9efc01e089c652bbb0a156",
    "mtp.layers.0.mlp.shared_expert_gate.weight":
        "acdd306d9996d356e4667a16ff63af6fef086fc7f570f1bc00bbe882a2a93f6e",
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight":
        "c5782dd92b479a1f8ef43d683fe9c917fd9e8fba81f23c1aae8817bfe4b476a6",
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight":
        "21a504876b5e7b788301557c68dfb757999430eba36c66ec8ab96a63604dbfd1",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight":
        "b71240036dea954e034d5ae04e097bcd7ac51cb28a44e4c93d198af3447278a1",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight":
        "40756866bf2a5e82c28e407f717b6fa07614d593f78e79c7ceb2cd4ffe6dda94",
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight":
        "2befafae6e944c1fc651fd73d4d527eb6f138a6c1679065b003128ee3150e859",
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight":
        "2db621f6a08b4085756fc69d60a4a1bd1d7fbf5d406456d2477075842efaae54",
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight":
        "afaf59f68e251939970ffdeacad0667dffc657a522bea62cdc44a52816fc49d6",
    "mtp.layers.0.self_attn.k_norm.weight":
        "8f924b0357cf936e14fc34518205d000e1423aa60626827df81b0804fae8cdfd",
    "mtp.layers.0.self_attn.k_proj.weight":
        "1a295dc2d6e633e808aeb19e00c97856a1817bbfcc83f5b610e2bfcacfcb8c38",
    "mtp.layers.0.self_attn.o_proj.weight":
        "9d2fcf0ac6ac19de87f638f710cfd1af6ad14141e75b7d3d5d29fbd06f374219",
    "mtp.layers.0.self_attn.q_norm.weight":
        "57957aa1e7c5a3ac628bff0ec5522ad55a39b8c0b8bd2d324ae640e3771617ef",
    "mtp.layers.0.self_attn.q_proj.weight":
        "5220aef3d288f40759f7356af9ef453e5ac0256a38749a9bb803a4e9db58150d",
    "mtp.layers.0.self_attn.v_proj.weight":
        "39cb12f5424e965fa71fb81db0ee230c701b7b1e0b72c36641fde7e0e41bff41",
    "mtp.pre_fc_norm_embedding.weight":
        "04c4a570850e06f2d8913da8220d54d4c7f87db6eb6d45480b938e8ba41d6a86",
    "mtp.pre_fc_norm_hidden.weight":
        "b933b44559879449b1baf2af8385c5c237347307e3fd452394870cb53e6eab6d",
}


def pinned():
    """The tensors' hashes are the pinned revision's: they apply when this run names that revision (or ModelScope's copy
    of it, which is checked the same way)."""
    return MODELSCOPE or REVISION == PINNED_REVISION


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def verify(out):
    """#327: the fetched tensors against SHA256 -> the names that are missing or wrong.  [] when they are right, or
    when there is nothing to check (no tensors/, or STRATA_MTP_REVISION names another revision).  A tensor's verdict
    is kept in tensors/verified.json by size and mtime, so a later run hashes only what changed."""
    tdir = os.path.join(out, "tensors")
    if not pinned() or not os.path.isdir(tdir):
        return []
    stamp_path = os.path.join(tdir, "verified.json")
    try:
        with open(stamp_path, encoding="utf-8") as f:
            stamps = json.load(f)
    except (OSError, ValueError):
        stamps = {}
    bad, good = [], {}
    for name, want in sorted(SHA256.items()):
        path = os.path.join(tdir, name + ".bin")
        if not os.path.exists(path):
            bad.append(name)
            continue
        st = os.stat(path)
        key = [st.st_size, st.st_mtime_ns, want]
        if stamps.get(name) != key and sha256_of(path) != want:
            bad.append(name)
            continue
        good[name] = key
    try:
        with open(stamp_path, "w", encoding="utf-8") as f:
            json.dump(good, f, indent=1)
    except OSError:
        pass
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["verify"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bad = verify(a.out)                             # offline: the hashes are the pinned revision's
    for name in bad:
        print("MTP tensor missing or corrupt: %s" % name, file=sys.stderr)
    sys.exit(BAD if bad else 0)


if __name__ == "__main__":
    main()
