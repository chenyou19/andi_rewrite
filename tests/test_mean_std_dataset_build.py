"""Exercise real LMDB publication/spectrum and interruption recovery on tiny data."""
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_build_recovery_publication_and_frozen_inputs(tmp_path, monkeypatch):
    b = importlib.import_module("prepare_mixed_meanstd_andi_histmatch")
    stage, output = tmp_path / "dataset.staging", tmp_path / "dataset"
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(b, "STAGE", stage)
    monkeypatch.setattr(b, "OUTPUT", output)
    monkeypatch.setattr(b, "REPORT", tmp_path / "report")
    monkeypatch.setattr(b, "EXPECTED_CASES", 3)
    monkeypatch.setattr(b.base, "SOURCE", tmp_path / "source")
    monkeypatch.setattr(b.base, "EXPECTED", {"train":2, "val":1})
    for name in ("data/mean_std_normalization.py", "scripts/prepare_mixed_meanstd_andi_histmatch.py"):
        target = tmp_path / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("frozen test input")
    shutil.copyfile(ROOT / "scripts/compute_lmdb_spectrum.py", tmp_path / "scripts/compute_lmdb_spectrum.py")
    for ds in b.base.SOURCES:
        p = tmp_path / f"outputs/datasets/{ds}_sri24_robust_iqr/build_report.json"
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text("{}")
    groups = [("train","fomo45k","fomo",[{"key":"00000000","z":0}]),
              ("train","mpi","mpi",[{"key":"00000001","z":0}]),
              ("val","oasis3","oasis",[{"key":"00000000","z":0}])]
    manifests = {}
    for split in ("train", "val"):
        path = b.base.manifest_path(b.base.SOURCE, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps({"key":rows[0]["key"], "source_dataset":ds,
                          "source_split":split, "source_key":rows[0]["key"]})+"\n"
                          for s,ds,c,rows in groups if s == split))
        manifests[split] = b.base.digest(path)
    monkeypatch.setattr(b.base, "case_groups", lambda: (groups, manifests))
    monkeypatch.setattr(b, "sessions", lambda ds: {"mpi" if ds=="mpi" else "oasis":{"statistics":[]}})
    monkeypatch.setattr(b.base, "case_paths", lambda ds, case: {m:Path(m) for m in b.base.MODALITIES})
    mask = np.zeros((1,3,128,128), dtype=bool); mask[:,:,32:96,32:96] = True
    values = np.full(mask.shape, -1, dtype=np.float32)
    values[mask] = np.tile(np.linspace(-2,3,4096,dtype=np.float32),3)
    monkeypatch.setattr(b, "source_slices", lambda *args: (values.copy(),mask.copy(),values.copy(),[],[]))
    monkeypatch.setattr(b, "verify_original", lambda *args: 0.0)
    curves = np.tile(np.linspace(-2,3,4097), (3,1)); ratios = np.ones(3)*3
    def reference(work):
        work.mkdir(parents=True,exist_ok=True)
        target = work / "reference.npz"
        if not target.exists(): np.savez_compressed(target, quantiles=curves, background_ratio=ratios)
        return curves, ratios, {"reference_sha256":b.base.digest(target)}
    monkeypatch.setattr(b, "build_reference", reference)
    export = stage / "fomo_meanstd_inputs"; export.mkdir(parents=True)
    contract = {"normalization":b.MEAN_STD_SPEC,"source_manifests":manifests,
                "normalizer_sha256":b.base.digest(tmp_path/'data/mean_std_normalization.py'),
                "source_report_sha256":b.base.digest(tmp_path/'outputs/datasets/fomo45k_sri24_robust_iqr/build_report.json')}
    b.dump(export / "contract.json", contract)
    np.savez_compressed(export/'fomo.npz', values=values, support=mask, case_id='fomo', z=np.asarray([0]))
    b.dump(export/'export_report.json', {"status":"PASS", "contract_sha256":b.base.digest(export/'contract.json'),
        "cases":[{"case_id":"fomo", "split":"train", "keys":["00000000"], "z":[0],
                  "cache_file":"fomo.npz", "cache_sha256":b.base.digest(export/'fomo.npz'),
                  "statistics":[], "raw_files":[], "original_reconstruction_max_abs_error":0.0}]})
    # Interrupt at spectrum generation after all case transactions were committed.
    real_run = b.subprocess.run
    def interrupted(*args, **kwargs):
        raise RuntimeError("simulated spectrum interruption")
    monkeypatch.setattr(b.subprocess, "run", interrupted)
    with pytest.raises(RuntimeError, match="simulated spectrum"):
        b.build()
    assert stage.exists() and not output.exists()
    monkeypatch.setattr(b.subprocess, "run", real_run)
    b.build()
    assert output.exists() and not stage.exists()
    report = json.loads((output/'build_report.json').read_text())
    assert report['status']=='PASS' and report['entries']=={'train':2,'val':1}
    meta = json.loads((output/'spectrum'/f'{b.SPECTRUM_NAME}.metadata.json').read_text())
    assert meta['source_serialized_values_sha256']==report['source_serialized_values_sha256']['train']
    assert meta['num_slices_used']+meta['num_slices_skipped']==2
    assert json.loads((tmp_path/'report/build_status.json').read_text())['status']=='PASS'
    b.build()  # Published output can be validated and reused.
    (tmp_path/'data/mean_std_normalization.py').write_text('changed input')
    with pytest.raises(ValueError, match='export contract differs'):
        b.build()
