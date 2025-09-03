def consistency_score(e_from_c:str, e_gt:str, c_from_e:str, c_gt:str) -> float:
    # TODO: 实现 C<->E 一致性打分（字符/AST 级别）
    return float(e_from_c == e_gt) * 0.5 + float(c_from_e == c_gt) * 0.5
