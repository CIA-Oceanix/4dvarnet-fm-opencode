from eval_sda_l96 import guidance_obs_indices


def test_slow_only_returns_first_no_positions():
    assert guidance_obs_indices(0, canonical_obs_j=2, NO=8) == list(range(8))


def test_none_or_covering_returns_unrestricted():
    assert guidance_obs_indices(None, canonical_obs_j=2, NO=8) is None
    assert guidance_obs_indices(2, canonical_obs_j=2, NO=8) is None
    assert guidance_obs_indices(3, canonical_obs_j=2, NO=8) is None


def test_partial_density_interleaves_per_node():
    # canonical_obs_j=2 means the 24D array is [X0..X7, (Y0,0 Y0,1), (Y1,0 Y1,1), ...];
    # asking for 1 fast var/node should pick the first (j=0) slot of every node.
    idx = guidance_obs_indices(1, canonical_obs_j=2, NO=8)
    assert idx == list(range(8)) + [8, 10, 12, 14, 16, 18, 20, 22]
