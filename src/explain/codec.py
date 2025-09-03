def code_from_sem(sem: dict) -> str:
    return "|".join(f"{k}:{sem[k]}" for k in ["ACT","OBJ","ATTR","LOC"])

def explain_from_sem(sem: dict) -> str:
    return f"NAV(act={sem['ACT']}, obj={sem['OBJ']}, attr={sem['ATTR']}, loc={sem['LOC']})"

def sem_from_code(code: str) -> dict:
    pairs = [t.split(":", 1) for t in code.split("|")]
    return {k: v for k, v in pairs}

def code_from_explain(e: str) -> str:
    inside = e[e.find("(")+1:e.rfind(")")]
    kv = dict(x.strip().split("=", 1) for x in inside.split(","))
    return f"ACT:{kv['act']}|OBJ:{kv['obj']}|ATTR:{kv['attr']}|LOC:{kv['loc']}"
