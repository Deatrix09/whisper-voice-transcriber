"""Validační testy pro Fix D (L04): NeMo temp cleanup + release + asr_based_vad=False.

NeMo není v testovacím prostředí nainstalováno -> stub `nemo.collections.asr`
s FakeNeuralDiarizer, který zapíše věrohodný RTTM. Reálné zůstávají:
preprocessing (librosa+soundfile), manifest, RTTM parsing, fallback.
"""
import os
import sys
import types
import uuid
from unittest.mock import patch

import pytest
import torch


def _install_nemo_stub(captured, fail_diarize=False):
    """Vytvoř hierarchii stub modulů nemo.collections.asr s fake diarizerem."""
    nemo = types.ModuleType("nemo")
    collections = types.ModuleType("nemo.collections")
    asr = types.ModuleType("nemo.collections.asr")

    class FakeNeuralDiarizer:
        def __init__(self, cfg):
            captured["cfg"] = cfg
            captured["instances"].append(self)

        def diarize(self):
            if fail_diarize:
                raise RuntimeError("fake diarize boom")
            pre = self._cfg_paths()[0]
            base = os.path.splitext(os.path.basename(pre))[0]
            out_dir = self._cfg_out_dir()
            rttm_dir = os.path.join(out_dir, "pred_rttms")
            os.makedirs(rttm_dir, exist_ok=True)
            # RTTM: >=8 sloupců, start=parts[3], dur=parts[4], speaker=parts[7]
            with open(os.path.join(rttm_dir, f"{base}.rttm"), "w") as f:
                f.write(f"SPEAKER {base} 1 0.0 1.0 <NA> <NA> SPEAKER_00 <NA>\n")
                f.write(f"SPEAKER {base} 1 1.5 2.0 <NA> <NA> SPEAKER_01 <NA>\n")

        def _cfg_paths(self):
            cfg = captured["cfg"]
            return [str(p) for p in cfg["diarizer"]["paths2audio_files"]]

        def _cfg_out_dir(self):
            return str(captured["cfg"]["diarizer"]["out_dir"])

    asr.models = types.SimpleNamespace(NeuralDiarizer=FakeNeuralDiarizer)
    collections.asr = asr
    nemo.collections = collections
    return {"nemo": nemo, "nemo.collections": collections, "nemo.collections.asr": asr}


@pytest.fixture(autouse=True)
def _clean_pred_rttms():
    """Izolace: kód hledá i 'jakýkoliv RTTM' v data/output/pred_rttms,
    zbytky z jiných testů by sebralo. Před i po testu uklidit."""
    import shutil
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "output")
    pred = os.path.join(root, "pred_rttms")
    if os.path.isdir(pred):
        for f in os.listdir(pred):
            p = os.path.join(pred, f)
            if os.path.isfile(p):
                os.remove(p)
    yield
    if os.path.isdir(pred) and not os.listdir(pred):
        try:
            os.rmdir(pred)
        except OSError:
            pass


@pytest.fixture
def nemo_config(app_config):
    app_config.diarization.enabled = True
    app_config.diarization.module = "nemo"
    app_config.performance.device = "cpu"
    return app_config


@pytest.fixture
def nemo_wav(tmp_path):
    from conftest import make_wav
    name = f"nemo_{uuid.uuid4().hex[:8]}.wav"
    return make_wav(tmp_path / name, seconds=2.0)


def _diarizer(nemo_config):
    from src.diarization.nemo_diarizer import NemoDiarizer
    return NemoDiarizer(nemo_config)


def _cfg_str(captured, *keys):
    """OmegaConf uzly -> plain str (os.path.* nebere AnyNode)."""
    node = captured["cfg"]
    for k in keys:
        node = node[k]
    return str(node)


def _test_rttm(captured, stem):
    return os.path.join(_cfg_str(captured, "diarizer", "out_dir"), "pred_rttms", f"{stem}_processed.rttm")


def test_success_parses_rttm_and_cleans_temp(nemo_config, nemo_wav):
    captured = {"cfg": None, "instances": []}
    stub = _install_nemo_stub(captured)
    with patch.dict(sys.modules, stub):
        diar = _diarizer(nemo_config)
        with patch.object(torch.cuda, "is_available", return_value=False), \
             patch("gc.collect") as gc_mock:
            results = diar.process(str(nemo_wav))
    # RTTM: 2 segmenty, gap 0.5 -> bez konsolidace (1.0 -> 1.5 gap = 0.5 <= 0.5? ano -> merge!)
    # 0.0-1.0 SPEAKER_00, 1.5-3.5 SPEAKER_01: různí speakeři -> 2 segmenty
    assert len(results) == 2
    assert results[0]["speaker"] == "SPEAKER_00"
    assert results[1]["speaker"] == "SPEAKER_01"
    # temp adresář (manifest dir) smazán
    manifest = str(captured["cfg"]["diarizer"]["manifest_filepath"])
    assert not os.path.exists(os.path.dirname(manifest))
    gc_mock.assert_called()
    # cleanup testovacího RTTM z data/output
    rttm = _test_rttm(captured, nemo_wav.stem)
    if os.path.exists(rttm):
        os.remove(rttm)


def test_asr_based_vad_disabled(nemo_config, nemo_wav):
    """NeMo nesmí dostat asr_based_vad=True (zbytečný Conformer download)."""
    captured = {"cfg": None, "instances": []}
    stub = _install_nemo_stub(captured)
    with patch.dict(sys.modules, stub):
        diar = _diarizer(nemo_config)
        diar.process(str(nemo_wav))
    cfg = captured["cfg"]
    assert cfg["asr"]["parameters"]["asr_based_vad"] is False
    # klíč 'asr' zůstal (žádný KeyError risk na žádné verzi NeMo)
    assert str(cfg["asr"]["model_path"]) == "stt_en_conformer_ctc_large"
    rttm = _test_rttm(captured, nemo_wav.stem)
    if os.path.exists(rttm):
        os.remove(rttm)


def test_release_empty_cache_on_cuda(nemo_config, nemo_wav):
    captured = {"cfg": None, "instances": []}
    stub = _install_nemo_stub(captured)
    with patch.dict(sys.modules, stub):
        diar = _diarizer(nemo_config)
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "empty_cache") as ec:
            diar.process(str(nemo_wav))
    ec.assert_called_once()
    rttm = _test_rttm(captured, nemo_wav.stem)
    if os.path.exists(rttm):
        os.remove(rttm)


def test_diarize_failure_fallback_and_temp_cleaned(nemo_config, nemo_wav):
    """Pád diarize() -> single-segment fallback se SPRÁVNOU délkou + temp uklizen."""
    captured = {"cfg": None, "instances": []}
    stub = _install_nemo_stub(captured, fail_diarize=True)
    with patch.dict(sys.modules, stub):
        diar = _diarizer(nemo_config)
        results = diar.process(str(nemo_wav))
    assert len(results) == 1
    assert results[0]["speaker"] == "SPEAKER_0"
    assert results[0]["duration"] == pytest.approx(2.0, abs=0.1)
    manifest = str(captured["cfg"]["diarizer"]["manifest_filepath"])
    assert not os.path.exists(os.path.dirname(manifest))
