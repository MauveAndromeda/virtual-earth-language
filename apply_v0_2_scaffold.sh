#!/usr/bin/env bash
set -euo pipefail
echo "🚀 Applying v0.2 scaffold (train loop + losses placeholders)..."

mkdir -p src/{train,aligners,objectives} experiments

# 简化版训练循环（占位，先跑 dry-run）
cat > experiments/train_v0_2.py <<'PY'
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
PY

# 解释性指标占位（后续细化）
cat > src/objectives/consistency.py <<'PY'
def consistency_score(e_from_c:str, e_gt:str, c_from_e:str, c_gt:str) -> float:
    # TODO: 实现 C<->E 一致性打分（字符/AST 级别）
    return float(e_from_c == e_gt) * 0.5 + float(c_from_e == c_gt) * 0.5
PY

cat > src/aligners/slot_ctc.py <<'PY'
def slot_alignment_score(msg:str, sem:dict) -> float:
    # TODO: CTC/单调对齐的真实实现；这里先返回1.0占位
    return 1.0
PY

# README 增加“v0.2 训练脚手架”段落
awk '1;/^## Next \(v0.2\)/{print "\n### Try the v0.2 scaffold\n```bash\nPYTHONPATH=. python experiments/train_v0_2.py  # dry-run\n```"}' README.md > README.md.tmp && mv README.md.tmp README.md

git add -A
git commit -m "feat(v0.2-scaffold): add minimal train loop and explainability placeholders"
echo "✅ v0.2 scaffold committed. Push when ready."
