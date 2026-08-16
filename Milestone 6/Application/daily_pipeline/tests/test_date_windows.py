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


def test_tomorrow_windows_use_today_as_eo_and_today_minus_five_for_era5():
    as_of = date(2026, 8, 13)
    target = date(2026, 8, 14)
    assert max(run_daily.eo_asof_dates_needed([target], config(), as_of=as_of)) == as_of
    assert max(run_daily.firms_label_dates_needed([target], config(), as_of=as_of)) == as_of
    assert max(run_daily.era5_days_needed([target], config(), as_of=as_of)) == date(2026, 8, 8)


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


def test_tomorrow_reuses_existing_final_before_preflight(monkeypatch):
    as_of = date(2026, 8, 13)
    target = as_of + run_daily.timedelta(days=1)
    artifact = {"labelDate": target.isoformat(), "featureCount": 86}
    events = []
    monkeypatch.setattr(run_daily, "pipeline_today", lambda _: as_of)
    monkeypatch.setattr(run_daily, "existing_final_artifact", lambda *_: artifact)
    monkeypatch.setattr(
        run_daily,
        "preflight_source_inventory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("preflight should be skipped")),
    )
    monkeypatch.setattr(run_daily, "emit_event", lambda *args, **kwargs: events.append((args, kwargs)))

    assert run_daily.cmd_all(_all_args(target), config()) == 0
    assert events[-1][0][1] == "succeeded"


def test_tomorrow_missing_inventory_is_unavailable_without_cloud_work(monkeypatch):
    as_of = date(2026, 8, 13)
    target = as_of + run_daily.timedelta(days=1)
    inventory = {
        "era5": {"required": 38, "available": 38, "missing": 0, "scheduled": 0, "pending": 0},
        "firms": {"required": 31, "available": 30, "missing": 1, "scheduled": 0, "pending": 0},
        "sentinel2": {"required": 7, "available": 7, "missing": 0, "scheduled": 0, "pending": 0},
        "sentinel5p": {"required": 31, "available": 31, "missing": 0, "scheduled": 0, "pending": 0},
    }
    events = []
    monkeypatch.setattr(run_daily, "pipeline_today", lambda _: as_of)
    monkeypatch.setattr(run_daily, "existing_final_artifact", lambda *_: None)
    monkeypatch.setattr(run_daily, "preflight_source_inventory", lambda *_args, **_kwargs: inventory)
    monkeypatch.setattr(run_daily, "emit_event", lambda *args, **kwargs: events.append((args, kwargs)))
    monkeypatch.setattr(
        run_daily,
        "cmd_download_for_labels",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("download must not run")),
    )
    monkeypatch.setattr(
        run_daily,
        "cmd_all_one",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("preprocessing must not run")),
    )

    assert run_daily.cmd_all(_all_args(target), config()) == 0
    unavailable = [event for event in events if event[0][1] == "unavailable"]
    assert len(unavailable) == 1
    assert unavailable[0][0][2] == "Tomorrow's data is not available yet."
    assert unavailable[0][1]["inventory"] == inventory


def test_tomorrow_complete_inventory_builds_without_download_scheduling(monkeypatch):
    as_of = date(2026, 8, 13)
    target = as_of + run_daily.timedelta(days=1)
    inventory = {
        key: {"required": 1, "available": 1, "missing": 0, "scheduled": 0, "pending": 0}
        for key in ("era5", "firms", "sentinel2", "sentinel5p")
    }
    built = []
    monkeypatch.setattr(run_daily, "pipeline_today", lambda _: as_of)
    monkeypatch.setattr(run_daily, "existing_final_artifact", lambda *_: None)
    monkeypatch.setattr(run_daily, "preflight_source_inventory", lambda *_args, **_kwargs: inventory)
    monkeypatch.setattr(run_daily, "emit_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run_daily,
        "cmd_download_for_labels",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("download must not run")),
    )
    monkeypatch.setattr(
        run_daily,
        "cmd_all_one",
        lambda _args, _cfg, label, **kwargs: built.append((label, kwargs["skip_download"])) or 0,
    )

    assert run_daily.cmd_all(_all_args(target), config()) == 0
    assert built == [(target, True)]
