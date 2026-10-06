"""Generate a dataset from a specification, then read it back.

    .venv/bin/python examples/04_generate_dataset.py [OUT_DIR]

The same as `plantbench generate examples/04_dataset.toml`.  Every run's
configuration, cost and linear features are in index.csv; its trajectory is in
runs/<run_id>.npz.  Run it twice: the second time there is nothing left to do.

A script that generates with more than one worker needs the `if __name__ == "__main__"`
guard at the bottom, because each worker process imports the script it was started from.
"""

import sys
from pathlib import Path

import numpy as np

from plantbench import datagen


def main() -> None:
    spec = Path(__file__).with_name("04_dataset.toml")
    out = datagen.generate(spec, out_dir=sys.argv[1] if len(sys.argv) > 1 else None,
                           workers=2, log=lambda msg: None)

    records = datagen.load_records(out)
    ok = [r for r in records if r["status"] == "ok"]
    print(f"{len(records)} runs in {out}: {len(ok)} ok, "
          f"{sum(r['status'] != 'ok' for r in records)} not")

    # The linear feature against the nonlinear outcome: how far the reactor temperature
    # still swings over the last 100 min of each run.
    rows = []
    for r in ok:
        tr = datagen.load_run(out, r["run_id"]).window(500.0, 600.0)
        rows.append((r["features"]["damping_ratio"], np.ptp(tr.y["T"])))
    rows.sort()
    for zeta, swing in rows[:3] + rows[-3:]:
        print(f"damping {zeta:6.3f}   late swing in T {swing:9.2e} K")


# Worker processes re-import this file; the guard keeps them from generating as well.
if __name__ == "__main__":
    main()
