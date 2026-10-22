from presto.config import load_config, save_config
from presto.seeding import make_rng


def test_override(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("a:\n  b: 1\n", encoding="utf-8")

    cfg = load_config(path, ["a.b=2", "a.c=[1,2]"])

    assert cfg["a"]["b"] == 2
    assert cfg["a"]["c"] == [1, 2]


def test_defaults_are_merged(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "base.yaml").write_text("a:\n  b: 1\n  c: 2\n", encoding="utf-8")
    (configs / "scenario.yaml").write_text("defaults: [base.yaml]\na:\n  c: 3\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert load_config(configs / "scenario.yaml") == {"a": {"b": 1, "c": 3}}


def test_save_config_creates_parent(tmp_path):
    path = tmp_path / "nested" / "config.yaml"
    save_config({"a": 1}, path)

    assert load_config(path) == {"a": 1}


def test_streams_independent():
    failures_a = make_rng(7, "failures").random(3)
    failures_b = make_rng(7, "failures").random(3)
    noise = make_rng(7, "noise").random(3)

    assert (failures_a == failures_b).all()
    assert not (failures_a == noise).all()
