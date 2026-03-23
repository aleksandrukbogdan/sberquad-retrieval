import argparse
import sys


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(prog="python -m lab.eval")
    parser.add_argument("--split", choices=("dev", "tune", "mini", "extra"), default="dev")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--rebuild-hyde", action="store_true")
    args = parser.parse_args(argv)
    if args.split == "mini":
        from lab.mini import check_baseline, run_mini, write_baseline

        report = run_mini()
        if args.write_baseline:
            path = write_baseline(report)
            print(path, flush=True)
        if args.check:
            check_baseline(report)
        if not args.write_baseline and not args.check:
            for name, block in report["systems"].items():
                print(name, {k: round(block[k], 4) for k in ("recall@5", "recall@10", "mrr@10", "ndcg@10")})
        return
    if args.split == "extra":
        from lab.extra import run_extra

        run_extra()
        return
    if args.split == "tune":
        from lab.run import run_tune

        run_tune(rebuild_hyde=args.rebuild_hyde)
        return
    from lab.run import run_dev

    run_dev(rebuild_hyde=args.rebuild_hyde)


if __name__ == "__main__":
    main()
