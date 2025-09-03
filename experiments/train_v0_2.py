import sys, pathlib, random
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from ontology.slots import SLOTS, VOCAB
from envs.referential import sample_semantics
from agents.speaker import Speaker
from agents.listener import Listener
from explain.codec import sem_from_code
from objectives.losses import success, topo_similarity, length_penalty

def train(steps=200, dry_run=True):
    spk, lst = Speaker(), Listener()
    succs, codes, sems = [], [], []
    for _ in range(steps):
        tgt = sample_semantics(); cand = [tgt, sample_semantics()]
        code, expl = spk.generate(tgt)
        pred = lst.act(code, cand)
        succs.append(success(pred, 0)); codes.append(code); sems.append(tgt)
        # TODO: 在此处接 PPO/Gumbel-ST、Cons/Align/Learn 的真实更新
    topo = topo_similarity(sems, codes, sem_from_code)
    print(f"[v0.2] steps={steps} Success={sum(succs)/len(succs):.3f} Topo~={topo:.3f}")
    if dry_run: print("dry-run mode (no parameter updates)")

if __name__ == "__main__":
    train(dry_run=True)
