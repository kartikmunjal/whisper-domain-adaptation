import importlib.util
from pathlib import Path
P=Path(__file__).resolve().parents[1]/"scripts"/"prepare_codec_tts_scaled_data.py"; S=importlib.util.spec_from_file_location("scaled",P); M=importlib.util.module_from_spec(S); S.loader.exec_module(M)
def test_scale_templates_are_unique_and_parameterized():
 assert len(M.SCALE_TEMPLATES)==8 and len(set(M.SCALE_TEMPLATES))==8 and all("{term}" in x for x in M.SCALE_TEMPLATES)
def test_common_controls_are_unique_and_not_parameterized():
 assert len(M.SCALE_COMMON)==10 and len(set(M.SCALE_COMMON))==10 and all("{term}" not in x for x in M.SCALE_COMMON)
