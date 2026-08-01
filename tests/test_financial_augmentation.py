import importlib.util
from pathlib import Path
import numpy as np

PATH=Path(__file__).resolve().parents[1]/"scripts"/"run_finetune.py"
SPEC=importlib.util.spec_from_file_location("run_finetune",PATH); MODULE=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)

def test_shoebox_rir_is_finite_and_has_reflections():
    rir=MODULE.shoebox_rir(np.random.default_rng(11)); assert np.isfinite(rir).all(); assert np.count_nonzero(rir)>1; assert np.max(np.abs(rir))<=1.0

def test_augmentation_is_clip_stable():
    audio=np.sin(np.linspace(0,100,16000,dtype=np.float32)); left=MODULE.augment_financial_audio(audio,11,"clip"); right=MODULE.augment_financial_audio(audio,11,"clip"); assert np.array_equal(left,right)

def test_augmentation_seed_changes_output():
    audio=np.sin(np.linspace(0,100,16000,dtype=np.float32)); assert not np.array_equal(MODULE.augment_financial_audio(audio,11,"clip"),MODULE.augment_financial_audio(audio,22,"clip"))
