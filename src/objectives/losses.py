from envs.referential import topo_distance

def success(pred_idx: int, gold_idx: int) -> float:
    return 1.0 if pred_idx == gold_idx else 0.0

def topo_similarity(sems: list[dict], codes: list[str], decode_sem) -> float:
    ok = 0
    for i in range(len(sems) - 1):
        d_sem = topo_distance(sems[i], sems[i+1])
        d_msg = topo_distance(sems[i], decode_sem(codes[i+1]))
        ok += int((d_sem == 0) == (d_msg == 0))
    return ok / max(1, len(sems) - 1)

def length_penalty(code: str) -> int:
    return len(code)
