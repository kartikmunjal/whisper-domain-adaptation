from __future__ import annotations
import importlib.util
from pathlib import Path
SCRIPT=Path(__file__).resolve().parents[1]/"scripts"/"analyze_synthetic_real_regression.py"; SPEC=importlib.util.spec_from_file_location("regression",SCRIPT); MODULE=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)
def test_cliffs_delta_direction(): assert MODULE.cliffs_delta([3,4],[1,2])==1.0 and MODULE.cliffs_delta([1,2],[3,4])==-1.0
def test_word_alignment_operation_counts():
    assert [x[0] for x in MODULE.word_alignment("revenue rose", "revenue rises")] == ["equal", "substitute"]
    assert [x[0] for x in MODULE.word_alignment("revenue rose", "revenue")] == ["equal", "delete"]
    assert [x[0] for x in MODULE.word_alignment("revenue", "revenue rose")] == ["equal", "insert"]
def test_word_classes():
    assert MODULE.word_class("the",{"revenue"})=="function"; assert MODULE.word_class("10",{"revenue"})=="numeric"; assert MODULE.word_class("revenue",{"revenue"})=="financial"
