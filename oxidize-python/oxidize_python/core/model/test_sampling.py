from oxidize_python.core.model.sampling import default_sampling_config, greedy, sample


def test_greedy_argmax() -> None:
    assert greedy([0.1, 0.9, 0.2]) == 1


def test_sample_nonempty() -> None:
    tok = sample([1.0, 2.0, 3.0], default_sampling_config())
    assert 0 <= tok < 3
