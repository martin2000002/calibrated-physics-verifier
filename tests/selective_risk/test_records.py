from __future__ import annotations

import json
import math

from physverify.schema.io import json_ready
from physverify.selective_risk.records import PointRow


def test_a_written_point_row_with_an_undefined_rate_reads_back() -> None:
    row = PointRow(
        k_sigma=2.0,
        alpha=0.02,
        n=150,
        decided=97,
        coverage=0.65,
        risk=0.18,
        risk_ci=(0.1, 0.27),
        false_fail_rate=0.18,
        missed_fail_rate=math.nan,
    )
    written = json.loads(json.dumps(json_ready(row.model_dump()), allow_nan=False))
    assert written["missed_fail_rate"] is None
    assert math.isnan(PointRow.model_validate(written).missed_fail_rate)
