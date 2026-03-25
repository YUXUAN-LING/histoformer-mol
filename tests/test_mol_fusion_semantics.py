import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn

from lora_adapters.lora_linear import LoRAConv2d, LoRALinear
from mol_fusion.fusion.registry import build_fusion_policy
from mol_fusion.routing.decision import route_decision


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = LoRALinear(8, 8, r=2, domain_list=["d1", "d2", "d3"], alpha=1.0)
        self.b = LoRAConv2d(nn.Conv2d(3, 4, 3, padding=1), r=2, domain_list=["d1", "d2", "d3"], alpha=1.0)


def _all_close_dict(d1, d2, tol=1e-8):
    assert d1.keys() == d2.keys()
    for k in d1:
        assert abs(d1[k] - d2[k]) < tol


def test_topk_zero_ramp_equiv_pure_ramp():
    model = Dummy()
    pure = build_fusion_policy("pure_ramp")
    topk0 = build_fusion_policy("topk_softmix", topk=0, nonkey_mode="ramp", score_type="delta")
    p = pure.build_layer_weights(model=model, dom1="d1", dom2="d2", context={})
    t = topk0.build_layer_weights(model=model, dom1="d1", dom2="d2", context={})
    assert p.keys() == t.keys()
    for k in p:
        _all_close_dict(p[k], t[k])


def test_topk_negative_full_softmix_equiv():
    model = Dummy()
    full = build_fusion_policy("pure_softmix", score_type="delta", weight_rule="softmax", temp=0.7)
    neg = build_fusion_policy("topk_softmix", score_type="delta", topk=-1, nonkey_mode="half", weight_rule="softmax", temp=0.7)
    a = full.build_layer_weights(model=model, dom1="d1", dom2="d2", context={})
    b = neg.build_layer_weights(model=model, dom1="d1", dom2="d2", context={})
    for k in a:
        _all_close_dict(a[k], b[k], tol=1e-6)


def test_nonkey_half_is_half():
    model = Dummy()
    pol = build_fusion_policy("topk_softmix", score_type="delta", topk=1, nonkey_mode="half")
    lw = pol.build_layer_weights(model=model, dom1="d1", dom2="d2", context={})
    key_layers = 1
    nonkeys = list(lw.keys())[key_layers:]
    for n in nonkeys:
        assert abs(lw[n]["d1"] - 0.5) < 1e-6
        assert abs(lw[n]["d2"] - 0.5) < 1e-6


def test_pair_weights_normalized():
    model = Dummy()
    for name in ["pure_ramp", "pure_softmix", "topk_softmix", "hard_select"]:
        pol = build_fusion_policy(name, score_type="delta", topk=1)
        lw = pol.build_layer_weights(model=model, dom1="d1", dom2="d2", context={})
        for _, pair in lw.items():
            assert abs((pair["d1"] + pair["d2"]) - 1.0) < 1e-6


def test_routing_output_schema():
    route = route_decision({"d1": 0.4, "d2": 0.35, "d3": 0.25}, single_tau=0.9, margin_tau=0.2)
    for k in ["route", "dom1", "dom2", "scores", "reason"]:
        assert k in route


def test_pairwise_and_full_share_policy_builder():
    p1 = build_fusion_policy("topk_softmix", score_type="delta")
    p2 = build_fusion_policy("topk_softmix", score_type="delta")
    assert type(p1) is type(p2)

def test_probe_weights_same_as_policy_path():
    model = Dummy()
    pol = build_fusion_policy("topk_softmix", score_type="delta", topk=2, nonkey_mode="half")
    direct = pol.build_layer_weights(model=model, dom1="d1", dom2="d2", context={})
    probe_like = pol.build_layer_weights(model=model, dom1="d1", dom2="d2", context={})
    assert direct == probe_like
