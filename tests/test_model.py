import torch

from vsa_bench.common import load_config
from vsa_bench.model import ModelConfig, VSAPathMoE, parameter_accounting


def test_production_parameter_target_and_sparsity():
    model = VSAPathMoE(ModelConfig())
    stats = parameter_accounting(model)
    assert 9_500_000 <= stats["total_parameters"] <= 10_500_000
    assert stats["active_fraction"] < 0.20
    assert stats["expert_parameters_per_route_all_layers"] == 131_328
    assert stats["fp32_expert_per_route_megabytes"] == 0.525312
    assert (
        stats["backbone_and_shared_parameters"]
        + stats["all_route_expert_parameters"]
        == stats["total_parameters"]
    )


def test_fused_and_separate_mlp_are_equivalent():
    torch.manual_seed(3)
    cfg = ModelConfig(vocab_size=320, block_size=16, d_model=32, n_layer=1, n_head=4, shared_hidden=24, expert_hidden=8, num_routes=4)
    model = VSAPathMoE(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 10))
    fused, _, _ = model(x, 2, kernel_mode="fused")
    separate, _, _ = model(x, 2, kernel_mode="separate")
    torch.testing.assert_close(fused, separate, rtol=1e-5, atol=1e-6)


def test_kv_cache_matches_full_forward():
    torch.manual_seed(4)
    cfg = ModelConfig(vocab_size=320, block_size=16, d_model=32, n_layer=2, n_head=4, shared_hidden=24, expert_hidden=8, num_routes=4)
    model = VSAPathMoE(cfg).eval()
    sequence = torch.randint(0, cfg.vocab_size, (1, 10))
    full, _, _ = model(sequence, 1)
    logits, _, cache = model(sequence[:, :5], 1, use_cache=True)
    pieces = [logits]
    for position in range(5, sequence.shape[1]):
        logits, _, cache = model(sequence[:, position:position+1], 1, past_key_values=cache, use_cache=True)
        pieces.append(logits)
    cached = torch.cat(pieces, dim=1)
    torch.testing.assert_close(full, cached, rtol=1e-5, atol=1e-6)


def test_matched_dense_controls_have_exact_active_and_total_budgets():
    sparse = parameter_accounting(VSAPathMoE(ModelConfig()))
    active_cfg = load_config("dense_active_cpu")
    total_cfg = load_config("dense_total_cpu")
    active = parameter_accounting(
        VSAPathMoE(ModelConfig.from_mapping(active_cfg["model"]))
    )
    total = parameter_accounting(
        VSAPathMoE(ModelConfig.from_mapping(total_cfg["model"]))
    )
    assert active["total_parameters"] == sparse["active_parameters_per_request"]
    assert total["total_parameters"] == sparse["total_parameters"]
    assert active_cfg["model"]["num_routes"] == 1
    assert total_cfg["model"]["num_routes"] == 1
