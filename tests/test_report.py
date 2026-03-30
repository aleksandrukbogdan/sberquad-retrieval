from pathlib import Path

from lab.report import dev_report_path, parse_main_table, recompute, table_matches


def test_report_replay():
    path = dev_report_path()
    assert path.exists(), "нет reports/dev.json"
    import json

    report = json.loads(path.read_text(encoding="utf-8"))
    fresh = recompute(report)
    for name, stats in fresh.items():
        for key, value in stats.items():
            assert abs(value - report["systems"][name][key]) < 1e-9, name


def test_readme_table_matches_report():
    import json

    readme = Path("README.md").read_text(encoding="utf-8")
    report = json.loads(dev_report_path().read_text(encoding="utf-8"))
    problems = table_matches(readme, report)
    assert not problems, problems
    parsed = parse_main_table(readme)
    assert set(parsed) >= set(report["systems"])
