from lab.mini import check_baseline, run_mini


def test_mini_matches_baseline():
    check_baseline(run_mini())
