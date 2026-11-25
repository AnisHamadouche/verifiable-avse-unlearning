import numpy as np
import torch
from avse_unlearning import make_model,infer_architecture

def test_cross_attention_forward_and_detect(tmp_path):
    torch.set_num_threads(2)
    model=make_model('cross_attention').eval()
    with torch.no_grad():
        x=model(torch.randn(1,32000),torch.rand(1,20,1,48,48))
    assert x.shape==(1,32000)
    path=tmp_path/'model.pt'
    torch.save(model.state_dict(),path)
    assert infer_architecture(path)=='cross_attention'
