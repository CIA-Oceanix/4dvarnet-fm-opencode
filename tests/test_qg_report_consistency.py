"""Internal consistency of the QG neural report's declarations and inputs.

Pure code/text inspection -- no ``experiments/`` data, so this runs in CI where
that gitignored directory does not exist.

The L96 counterpart (``test_l96_report_consistency.py``) guards rows added to a
published table by hand. QG's failure was one step earlier: the table's four
neural rows were generated from a JSON that existed only in the worktree that
trained the models and was never committed, so regenerating the report anywhere
else silently dropped them. These tests hold the inputs in place.
"""
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "reports/qg/generate_qg_neural_report.py"
OUT = ROOT / "reports/qg/outputs/qg_neural_report.md"
EVAL = ROOT / "eval_qg_neural_s0_s1.py"
CONSOLIDATE = ROOT / "scripts/consolidate_qg_archive.py"
NEURAL_JSON = (ROOT / "reports/qg/outputs/qg_neural_s0_s1_cross_scenario"
               / "results_lag5_noise0.05_bias0.1.json")


@pytest.fixture(scope="module")
def gen_src() -> str:
    return GEN.read_text()


@pytest.fixture(scope="module")
def eval_src() -> str:
    return EVAL.read_text()


def _block(src: str, pattern: str) -> str:
    m = re.search(pattern, src, re.S)
    assert m, f"could not find {pattern!r}"
    return m.group(1)


def generator_schemes(src: str) -> list[str]:
    return re.findall(r'\("(Q\d+)",', _block(src, r"NEURAL_SCHEMES = \[(.*?)\n\]"))


def eval_schemes(src: str) -> dict[str, str]:
    blk = _block(src, r"SCHEMES = \{(.*?)\n\}")
    return dict(re.findall(r'"(Q\d+)":\s*\{"run":\s*"([^"]+)"', blk))


def tracked(path: Path) -> bool:
    out = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)],
                         cwd=ROOT, capture_output=True)
    return out.returncode == 0


class TestGeneratorAndEvalAgree:
    def test_every_published_scheme_is_evaluated(self, gen_src, eval_src):
        missing = [s for s in generator_schemes(gen_src) if s not in eval_schemes(eval_src)]
        assert not missing, (
            f"schemes the report publishes that no eval scheme produces: {missing}")

    def test_every_evaluated_scheme_is_published(self, gen_src, eval_src):
        orphans = [s for s in eval_schemes(eval_src) if s not in generator_schemes(gen_src)]
        assert not orphans, (
            f"schemes evaluated but absent from the report's table: {orphans}")

    def test_eval_schemes_name_a_run_not_a_checkpoint_path(self, eval_src):
        """Architecture must come from the run, not from literals at the call
        site -- the drift this indirection removed."""
        blk = _block(eval_src, r"SCHEMES = \{(.*?)\n\}")
        assert "param_dim" not in blk, (
            "SCHEMES re-declares architecture; read it from the run instead "
            "(evaluation.qg_runs.architecture)")
        assert '"ckpt"' not in blk, "SCHEMES hardcodes a checkpoint path"

    def test_cond_mode_stays_an_eval_time_choice(self, eval_src):
        """It is deliberately NOT read from the training config: the schemes
        train with "true"/"noisy" and evaluate with "scenario", and taking the
        training value would make the S1 comparison meaningless."""
        blk = _block(eval_src, r"SCHEMES = \{(.*?)\n\}")
        assert blk.count("cond_mode") == len(eval_schemes(eval_src))


class TestReportInputsAreReproducible:
    def test_neural_results_json_is_tracked(self):
        """The whole point: an input that lives only in one worktree makes the
        published rows unregenerable everywhere else."""
        assert NEURAL_JSON.is_file(), f"{NEURAL_JSON} is missing"
        assert tracked(NEURAL_JSON), f"{NEURAL_JSON} is not tracked in git"

    def test_da_results_json_are_tracked(self, gen_src):
        methods = re.findall(r'\("[^"]+",\s*"([^"]+)"\)',
                             _block(gen_src, r"DA_METHODS = \[(.*?)\]"))
        for scenario_dir in re.findall(r'load_da_summary\(out_root, "([^"]+)"', gen_src):
            for m in methods:
                p = ROOT / "reports/qg/outputs" / scenario_dir / f"{m}.json"
                assert tracked(p), f"{p} is not tracked in git"

    def test_neural_json_covers_every_published_scheme(self, gen_src):
        results = json.loads(NEURAL_JSON.read_text())["results"]
        missing = [s for s in generator_schemes(gen_src) if s not in results]
        assert not missing, f"schemes with no numbers in the tracked JSON: {missing}"

    def test_neural_json_carries_both_scenarios(self):
        for scheme, r in json.loads(NEURAL_JSON.read_text())["results"].items():
            assert {"S0", "S1"} <= set(r), f"{scheme} is missing a scenario"


class TestConsolidationScriptTracksTheEval:
    def test_it_reads_the_runs_from_the_eval_script(self, eval_src):
        """Parsed, not duplicated -- so adding a scheme cannot leave the archive
        audit behind."""
        src = CONSOLIDATE.read_text()
        assert "EVAL_SCRIPT.read_text()" in src
        for run in eval_schemes(eval_src).values():
            assert run not in src, f"{run} is hardcoded in the consolidation script"


@pytest.mark.skipif(not OUT.exists(), reason="checked-in report not present")
class TestPublishedTableMatchesItsInput:
    def test_published_neural_rows_match_the_tracked_json(self, gen_src):
        """Every neural EV in the checked-in markdown must be the value the
        tracked JSON holds, to 4 decimals -- the report and its input drifting
        apart is how a table becomes historical rather than reproducible."""
        results = json.loads(NEURAL_JSON.read_text())["results"]
        labels = dict(re.findall(r'\("(Q\d+)",\s*"([^"]+)"', _block(
            gen_src, r"NEURAL_SCHEMES = \[(.*?)\n\]")))
        published = {}
        for line in OUT.read_text().splitlines():
            m = re.match(r"^\|\s*([^|]+?)\s*\|((?:\s*\**\*?[-\d.]+\**\s*\|){4})\s*$", line)
            if m:
                cells = [c.strip().strip("*") for c in m.group(2).split("|")[:4]]
                published[m.group(1)] = cells
        for scheme, label in labels.items():
            assert label in published, f"no published row labelled {label!r}"
            want = [f"{results[scheme][case][field]['pooled_ev']:.4f}"
                    for case in ("S0", "S1") for field in ("psi", "q")]
            # table column order is S0 psi, S0 q, S1 psi, S1 q
            assert published[label] == want, (
                f"{label}: published {published[label]} vs JSON {want}")
