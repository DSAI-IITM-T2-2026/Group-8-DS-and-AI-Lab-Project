from datetime import date
import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "utils"
sys.path.insert(0, str(UTILS))
# Date-window helpers do not need preprocessing libraries; stub the two eager
# imports so this lightweight contract test runs in the API test environment.
preprocess_pkg = types.ModuleType("preprocess")
preprocess_pkg.__path__ = []
stage_module = types.ModuleType("preprocess.build_stage_c_day")
stage_module.run_stage_c_pipeline = lambda *args, **kwargs: None
export_module = types.ModuleType("preprocess.export_inference_day")
export_module.export_champion_day = lambda *args, **kwargs: None
export_module.validate_champion_artifact = lambda *args, **kwargs: []
final_module = types.ModuleType("preprocess.final_artifact")
final_module.existing_final_artifact = lambda *args, **kwargs: None
module_stubs = {
    "preprocess": preprocess_pkg,
    "preprocess.build_stage_c_day": stage_module,
    "preprocess.export_inference_day": export_module,
    "preprocess.final_artifact": final_module,
}
saved_modules = {name: sys.modules.get(name) for name in module_stubs}
sys.modules.update(module_stubs)
spec = importlib.util.spec_from_file_location("run_daily", ROOT / "run_daily.py")
run_daily = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(run_daily)
for name, previous in saved_modules.items():
    if previous is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = previous


def config():
    return {
        "task": {"lookback_days": 30, "history_days": 7, "era5_lag_days": 5, "lead_days": 1},
    }


def test_causal_windows_at_supported_boundary():
    target = date(2019, 1, 1)
    assert run_daily.eo_asof_date(target) == date(2018, 12, 31)
    assert run_daily.era5_feature_end(target) == date(2018, 12, 26)
    assert run_daily.firms_label_dates_needed([target], config(), as_of=target)[0] == date(2018, 12, 2)
    assert run_daily.firms_label_dates_needed([target], config(), as_of=target)[-1] == date(2018, 12, 31)


def test_era5_window_includes_history_and_never_passes_d_minus_six():
    target = date(2025, 8, 1)
    days = run_daily.era5_days_needed([target], config(), as_of=target)
    assert days[0] == date(2025, 6, 19)
    assert days[-1] == date(2025, 7, 26)


def test_future_labels_are_bounded_by_as_of():
    target = date(2026, 8, 20)
    as_of = date(2026, 8, 13)
    assert max(run_daily.eo_asof_dates_needed([target], config(), as_of=as_of)) == as_of
    assert max(run_daily.era5_days_needed([target], config(), as_of=as_of)) == date(2026, 8, 7)


def _all_args(target):
    return types.SimpleNamespace(
        label_date=target,
        date=None,
        start_date=None,
        end_date=None,
        force=False,
        local_only=False,
    )


def test_all_reuses_valid_final_before_raw_inventory(monkeypatch):
    target = date(2026, 8, 12)
    artifact = {"labelDate": target.isoformat(), "featureCount": 86}
    events = []
    monkeypatch.setattr(run_daily, "pipeline_today", lambda _: target)
    monkeypatch.setattr(run_daily, "existing_final_artifact", lambda *_: artifact)
    monkeypatch.setattr(
        run_daily,
        "cmd_download_for_labels",
        lambda *_: (_ for _ in ()).throw(AssertionError("raw inventory should be skipped")),
    )
    monkeypatch.setattr(run_daily, "emit_event", lambda *args, **kwargs: events.append((args, kwargs)))

    assert run_daily.cmd_all(_all_args(target), config()) == 0
    assert events[-1][0][:3] == (
        "completed",
        "succeeded",
        "Existing prediction data is ready.",
    )


def test_all_falls_through_when_final_is_missing(monkeypatch):
    target = date(2026, 8, 12)
    calls = {"download": 0, "build": 0}
    monkeypatch.setattr(run_daily, "pipeline_today", lambda _: target)
    monkeypatch.setattr(run_daily, "existing_final_artifact", lambda *_: None)
    monkeypatch.setattr(run_daily, "emit_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_daily,
        "cmd_download_for_labels",
        lambda *_: calls.__setitem__("download", calls["download"] + 1) or 0,
    )
    monkeypatch.setattr(
        run_daily,
        "cmd_all_one",
        lambda *_args, **_kwargs: calls.__setitem__("build", calls["build"] + 1) or 0,
    )

    assert run_daily.cmd_all(_all_args(target), config()) == 0
    assert calls == {"download": 1, "build": 1}
