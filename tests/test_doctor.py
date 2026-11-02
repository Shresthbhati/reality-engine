import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("doctor", Path(__file__).resolve().parent.parent / "scripts" / "doctor.py")
doctor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(doctor)


def test_all_checks_report_a_valid_status():
    rows = doctor.run_checks()
    assert {r["check"] for r in rows} >= {"python", "venv", "torch", "gpu", "colmap", "node", "npm", "db"}
    assert all(r["status"] in {"OK", "WARN", "FAIL"} for r in rows)


def test_json_output_and_exit_code(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'x' / 'app.db').as_posix()}")
    assert doctor.main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_unwritable_db_fails(monkeypatch, tmp_path):
    f = tmp_path / "afile"
    f.write_text("x")  # parent of the db path is a file, not a directory -> cannot create a probe inside it
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(f / 'app.db').as_posix()}")
    assert doctor.check_db()[0] == "FAIL"
    assert doctor.main([]) == 1
