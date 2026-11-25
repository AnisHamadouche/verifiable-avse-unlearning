from unlearning_audit import auc_binary

def test_auc():
    assert auc_binary([0.9,0.8],[0.1,0.2])==1
    assert auc_binary([0.5],[0.5])==0.5
    assert auc_binary([0.1],[0.9])==0
