from pymegdec.stimulus_latent_autoencoder import _split_source_participants


def test_split_source_participants_seeded_random_is_deterministic():
    kwargs = {"strategy": "seeded_random", "seed": 7, "anchor": 3}

    train_a, validation_a = _split_source_participants((1, 2, 4, 5, 6, 8), 2, **kwargs)
    train_b, validation_b = _split_source_participants((1, 2, 4, 5, 6, 8), 2, **kwargs)

    assert train_a == train_b
    assert validation_a == validation_b
    assert len(validation_a) == 2
    assert set(train_a).isdisjoint(validation_a)
    assert tuple(sorted((*train_a, *validation_a))) == (1, 2, 4, 5, 6, 8)


def test_split_source_participants_seeded_random_changes_with_anchor():
    source = (1, 2, 4, 5, 6, 8, 9, 10)

    _train_a, validation_a = _split_source_participants(source, 3, strategy="seeded_random", seed=11, anchor=3)
    _train_b, validation_b = _split_source_participants(source, 3, strategy="seeded_random", seed=11, anchor=4)

    assert validation_a != validation_b


def test_split_source_participants_round_robin_uses_seed_and_anchor_offset():
    train, validation = _split_source_participants((1, 2, 4, 5, 6), 2, strategy="round_robin", seed=0, anchor=3)

    assert validation == (5, 6)
    assert train == (1, 2, 4)
