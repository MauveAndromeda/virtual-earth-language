from explain.codec import code_from_sem, explain_from_sem, code_from_explain

SEM = {"ACT":"GO","OBJ":"TRI","ATTR":"RED","LOC":"L01"}

def test_roundtrip():
    e = explain_from_sem(SEM)
    c1 = code_from_sem(SEM)
    c2 = code_from_explain(e)
    assert c1 == c2
