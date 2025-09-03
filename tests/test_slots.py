from ontology.slots import SLOTS, VOCAB

def test_slots_vocab_nonempty():
    assert set(SLOTS) == {"ACT","OBJ","ATTR","LOC"}
    for k in SLOTS:
        assert len(VOCAB[k]) >= 2
