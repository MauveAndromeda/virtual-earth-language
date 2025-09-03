import os, sys, random
from pathlib import Path

# Add src/ to sys.path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from ontology.slots import SLOTS, VOCAB
from envs.referential import sample_semantics
from agents.speaker import Speaker
from agents.listener import Listener
from explain.codec import sem_from_code
from objectives.losses import success, topo_similarity, length_penalty

random.seed(0)
spk, lst = Speaker(), Listener()
sems, codes, succs = [], [], []

for _ in range(100):
    target = sample_semantics()
    distractor = sample_semantics()
    cands = [target, distractor]
    msg_code, msg_expl = spk.generate(target)
    pred = lst.act(msg_code, cands)
    sems.append(target); codes.append(msg_code)
    succs.append(success(pred, 0))

topo = topo_similarity(sems, codes, sem_from_code)
avg_succ = sum(succs)/len(succs)
avg_len = sum(length_penalty(c) for c in codes)/len(codes)

print("=== MWE Metrics ===")
print(f"Success={avg_succ:.3f}  Topo~={topo:.3f}  AvgLen={avg_len:.1f}")
print("\n=== Samples (C ↔ E) ===")
for i in range(5):
    s = sems[i]
    print(f"{i+1}. CODE=ACT:{s['ACT']}|OBJ:{s['OBJ']}|ATTR:{s['ATTR']}|LOC:{s['LOC']}")
