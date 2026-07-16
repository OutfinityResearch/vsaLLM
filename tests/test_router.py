import numpy as np

from vsa_bench.router import RouterConfig, fit_router


def test_router_is_deterministic_and_bounded():
    prefixes = [f"prefix number {i} with a small story" for i in range(32)]
    cfg = RouterConfig(dimension=64, prefix_bytes=64, role_period=8, product_k=2, kmeans_iterations=3, seed=9)
    a, _ = fit_router(prefixes, cfg)
    b, _ = fit_router(prefixes, cfg)
    routes_a = [a.route(text) for text in prefixes]
    routes_b = [b.route(text) for text in prefixes]
    assert routes_a == routes_b
    assert all(0 <= route < 4 for route in routes_a)


def test_route_margin_returns_integer():
    prefixes = [f"different prefix {i} for routing" for i in range(16)]
    router, _ = fit_router(prefixes, RouterConfig(dimension=64, product_k=2, role_period=8, seed=7))
    route, margin = router.route_with_margin(prefixes[0])
    assert 0 <= route < 4
    assert isinstance(margin, int)
