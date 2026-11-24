"""Names read letter by letter: lexicon matching, the engines' switch to the letter model, the
synthetic name strips and the training task."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from scanlake_core.names import EMPTY, NAME_CHARS, NameScorer, build_lexicon, lexicon
from tests.conftest import ASSETS

C = len(NAME_CHARS) + 1


def _spelling(*texts: str, p: float = 0.95, t_steps: int = 64) -> np.ndarray:
    """[T, N, C] log-probabilities of a model that reads each text (3 blanks, then 2 steps per
    character and a blank after each), p sure of every step."""
    out = np.full((t_steps, len(texts), C), np.log((1 - p) / (C - 1)), dtype=np.float32)
    for n, text in enumerate(texts):
        seq = [0, 0, 0]
        for ch in text:
            seq += [NAME_CHARS.index(ch) + 1] * 2 + [0]
        seq += [0] * (t_steps - len(seq))
        for t, i in enumerate(seq[:t_steps]):
            out[t, n, i] = np.log(p)
    return out


@pytest.fixture(scope="module")
def scorer() -> NameScorer:
    return NameScorer(lexicon())


def _read(scorer: NameScorer, *texts: str) -> list[tuple[str, float]]:
    probs = np.exp(scorer.scores(_spelling(*texts)))
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-4)
    classes = scorer.lexicon.classes
    return [(classes[int(p.argmax())], float(p.max())) for p in probs]


def test_lexicon() -> None:
    lex = lexicon()
    assert lex.classes[-1] == EMPTY and len(set(lex.classes)) == len(lex.classes)
    assert {"Copper", "Tin", "Ice", "Inert Materials", "Quantanium", "Beradom"} <= set(lex.classes)
    assert all(c in NAME_CHARS for form in lex.forms for c in form)
    assert "RAW QUANTAINIUM (RAW)" in lex.forms  # the game's spelling
    for form, (a, b), k in zip(lex.forms, lex.cores, lex.form_class, strict=True):
        assert form[a:b].replace(" ", "").isalpha() and lex.classes[k] != EMPTY


def test_whole_names_with_and_without_suffix_or_prefix(scorer: NameScorer) -> None:
    texts = ["TIN (ORE)", "COPPER (ORE)", "COPPER", "RAW ICE", "ICE", "RAW SILICON",
             "INERT MATERIALS", "HEPHAESTANITE (ORE)", "SADARYX", "ALUMINUM (ORE)"]  # fmt: skip
    want = ["Tin", "Copper", "Copper", "Ice", "Ice", "Silicon", "Inert Materials",
            "Hephaestanite", "Sadaryx", "Aluminium"]  # fmt: skip
    for (got, confidence), cls in zip(_read(scorer, *texts), want, strict=True):
        assert got == cls and confidence > 0.95


def test_names_clipped_by_the_scrolling_window(scorer: NameScorer) -> None:
    for got, confidence in _read(scorer, "ANTAINIUM (RAW", "QUANTAINIUM (RA", "UANTAINIUM (RA"):
        assert got == "Quantanium" and confidence > 0.9


def test_an_empty_row_is_not_a_clipped_name(scorer: NameScorer) -> None:
    ((got, confidence),) = _read(scorer, "")
    assert got == EMPTY and confidence > 0.9


def test_unknown_text_comes_out_unsure(scorer: NameScorer) -> None:
    # garbage, and a misread letter the model is sure of: no confident wrong name
    for _, confidence in _read(scorer, "XQZJK WVX", "ACRICIUM (ORE)"):
        assert confidence < 0.1
    # a name the lexicon lacks reads as unsure instead of as its nearest neighbour
    small = NameScorer(build_lexicon({"Iron": ["IRON"], "Copper": ["COPPER"]}, [" (ORE)"], []))
    ((_, confidence),) = _read(small, "TIN (ORE)")
    assert confidence < 0.5


def test_a_letter_the_model_cannot_read_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot read"):
        NameScorer(build_lexicon({"Tin": ["TIN"]}, [], []), alphabet="ABC")


# -- engines ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def letter_models(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A models folder: the shipped digit and word models plus an untrained letter model."""
    pytest.importorskip("torch")
    if not (ASSETS / "models" / "digit_crnn.onnx.data").exists():
        pytest.skip("assets not fetched")
    from scanlake_models import checkpoint
    from scanlake_training.export import export_onnx
    from scanlake_training.names import NameTask
    from scanlake_training.tasks import CrnnTask

    folder = tmp_path_factory.mktemp("models")
    for name in ("digit_crnn", "word_model2k"):
        for suffix in (".onnx", ".onnx.data", ".json"):
            shutil.copyfile(ASSETS / "models" / f"{name}{suffix}", folder / f"{name}{suffix}")
    digits = CrnnTask()  # PyTorch weights for the digit model too (untrained: only shapes matter)
    checkpoint.save(digits.model(), folder / "digit_crnn", digits.meta({}))
    task = NameTask()
    checkpoint.save(task.model(), folder / "name_crnn", task.meta({}))
    export_onnx(folder / "name_crnn", folder / "name_crnn")
    return folder


def test_engines_read_names_with_the_letter_model(letter_models: Path) -> None:
    from scanlake_inference.metadata import ModelFiles
    from scanlake_inference.ort_backend import OrtEngine
    from scanlake_inference.torch_backend import TorchEngine

    files = ModelFiles(letter_models)
    assert files.reads_names and files.names_model == "name_crnn"
    batch = np.random.default_rng(0).random((3, 1, 32, 256), dtype=np.float32)
    for engine in (OrtEngine(files, "cpu"), TorchEngine(files, "cpu")):
        assert engine.word_classes == lexicon().classes
        scores = engine.words(batch)
        assert scores.shape == (3, len(lexicon().classes))
        assert np.allclose(np.exp(scores).sum(axis=1), 1, atol=1e-4)
    meta = json.loads((letter_models / "name_crnn.json").read_text(encoding="utf-8"))
    assert meta["modelType"] == "name_crnn" and meta["charClasses"] == NAME_CHARS


def test_without_a_letter_model_the_word_classifier_reads_names(tmp_path: Path) -> None:
    from scanlake_inference.metadata import ModelFiles

    files = ModelFiles(tmp_path)
    assert not files.reads_names and files.names_model == "word_model2k"


def test_captures_record_the_letter_model(
    letter_models: Path, word_models: Path, tmp_path: Path
) -> None:
    from scanlake_desktop.assets import AssetSet

    assets = AssetSet(ASSETS, tmp_path)
    shutil.copytree(word_models, tmp_path / "models")
    assert assets.models_dir == tmp_path / "models"
    assert assets.models_in_use() == ("digit_crnn", "word_model2k")
    for path in letter_models.glob("name_crnn.*"):
        shutil.copyfile(path, tmp_path / "models" / path.name)
    assert assets.models_in_use() == ("digit_crnn", "name_crnn")
    assert assets.model_versions().split(",")[1].startswith("name_crnn@")


# -- training -----------------------------------------------------------------------------


def _glyph_set(folder: Path) -> Path:
    """Letters drawn with OpenCV's font, as a glyph folder (stems as in scanlake_training.names)."""
    from scanlake_training.names import STEMS

    folder.mkdir(parents=True)
    for ch in NAME_CHARS.replace(" ", ""):
        g = np.zeros((16, 11), np.uint8)
        cv2.putText(g, ch, (0, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1)
        cv2.imwrite(str(folder / f"{STEMS.get(ch, ch)}_cv_0.png"), g)
    return folder


def test_synthetic_name_strips(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from scanlake_core.preprocess import PREPROCESS_VERSION, resize_pad
    from scanlake_training.data import load_gray
    from scanlake_training.names import generate_names, load_glyphs, name_samples

    glyphs = load_glyphs(_glyph_set(tmp_path / "glyphs"))
    assert set(glyphs) == set(NAME_CHARS) - {" "}
    rng = np.random.default_rng(1)
    backgrounds = [rng.integers(0, 200, (40, 300), dtype=np.uint8) for _ in range(3)]
    out = tmp_path / "synth"
    assert generate_names(out, 60, [(glyphs, 1.0)], backgrounds, seed=3, empty_share=0.1) == 60
    samples = name_samples(out)
    assert len(samples) == 60
    labels = [s.label for s in samples]
    assert "" in labels and any(" (" in label for label in labels)
    assert all(label == label.strip() for label in labels)
    for s in samples[:10]:
        assert resize_pad(load_gray(s.path), version=PREPROCESS_VERSION).shape == (32, 256)
    again = tmp_path / "again"
    generate_names(again, 60, [(glyphs, 1.0)], backgrounds, seed=3, empty_share=0.1)
    assert sorted(p.name for p in out.iterdir()) == sorted(p.name for p in again.iterdir())


def test_a_clipped_strip_is_labelled_with_what_shows() -> None:
    from scanlake_training.names import compose_name

    glyphs = {c: [np.full((14, 8), 255, np.uint8)] for c in NAME_CHARS if c != " "}
    backgrounds = [np.zeros((60, 400), np.uint8)]
    clipped = 0
    for seed in range(200):
        image, label = compose_name("QUANTAINIUM (RAW)", glyphs, backgrounds, random.Random(seed))
        assert image.ndim == 2 and label and label in "QUANTAINIUM (RAW)"
        clipped += label != "QUANTAINIUM (RAW)"
    assert 10 < clipped < 100  # about a fifth are clipped


def test_the_name_task_starts_from_the_digit_backbone(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from scanlake_models import checkpoint
    from scanlake_training.names import NameTask
    from scanlake_training.tasks import CrnnTask

    digits = CrnnTask()
    old = digits.model()
    checkpoint.save(old, tmp_path / "digit_crnn", digits.meta({}))
    task = NameTask()
    model = task.model()
    task.init_from(model, tmp_path / "digit_crnn")
    state = model.state_dict()
    assert torch.equal(state["cnn.0.weight"], old.state_dict()["cnn.0.weight"])
    assert state["fc.weight"].shape[0] == len(NAME_CHARS) + 1
    assert task.meta({})["modelType"] == "name_crnn" and "formats" not in task.meta({})


def test_name_samples_skip_foreign_labels(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from scanlake_training.names import name_samples

    for name in ("TIN (ORE)_1.png", "blank_2.png", "12.5_3.png", "tin_4.png", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    assert {(s.path.name, s.label) for s in name_samples(tmp_path)} == {
        ("TIN (ORE)_1.png", "TIN (ORE)"),
        ("blank_2.png", ""),
    }
