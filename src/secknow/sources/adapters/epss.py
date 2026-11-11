from __future__ import annotations

import csv
import gzip
from collections.abc import Iterator
from pathlib import Path

from secknow.domain.models import SourceRecord
from secknow.sources.adapters.base import AdapterContext, BaseSourceAdapter


class EpssCsvAdapter(BaseSourceAdapter):
    adapter_name = "epss_csv"
    suffixes = (".csv", ".gz")
    input_format = None

    def iter_records(self, path: Path, context: AdapterContext) -> Iterator[SourceRecord]:
        for file in self.files(path):
            opener = gzip.open if file.suffix.lower() == ".gz" else open
            with opener(file, "rt", encoding="utf-8-sig", newline="") as stream:
                first = stream.readline()
                model_version = ""
                score_date = ""
                if first.startswith("#"):
                    for field in first[1:].strip().split(","):
                        key, _, value = field.partition(":")
                        if key.strip() == "model_version":
                            model_version = value.strip()
                        elif key.strip() == "score_date":
                            score_date = value.strip()
                else:
                    stream.seek(0)
                reader = csv.DictReader(stream)
                required = {"cve", "epss", "percentile"}
                if not reader.fieldnames or not required.issubset(reader.fieldnames):
                    raise ValueError("EPSS CSV lacks cve, epss, or percentile columns")
                for row in reader:
                    cve_id = row.get("cve")
                    if not cve_id:
                        continue
                    score = float(row["epss"])
                    percentile = float(row["percentile"])
                    if not 0 <= score <= 1 or not 0 <= percentile <= 1:
                        raise ValueError(f"invalid EPSS probability for {cve_id}")
                    yield self.record(
                        context=context,
                        path=file,
                        native_id=cve_id,
                        title=f"EPSS score for {cve_id}",
                        body=f"EPSS probability {score:.8g}; percentile {percentile:.8g}.",
                        metadata={
                            "epss": score,
                            "percentile": percentile,
                            "model_version": model_version,
                            "score_date": score_date,
                        },
                    )


EPSSCSVAdapter = EpssCsvAdapter
