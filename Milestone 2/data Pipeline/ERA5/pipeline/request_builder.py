"""Build CDS API request payloads for monthly downloads."""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.config import PipelineConfig, days_for_month


@dataclass(frozen=True)
class DownloadRequest:
    year: str
    month: str
    output_path: str
    payload: dict

    @property
    def key(self) -> str:
        return f"{self.year}_{self.month}"


def build_monthly_requests(config: PipelineConfig) -> list[DownloadRequest]:
    requests: list[DownloadRequest] = []

    for year in config.years:
        for month in [f"{m:02d}" for m in range(1, 13)]:
            outfile = config.raw_file(year, month)
            payload = {
                "product_type": config.product_type,
                "variable": config.variables,
                "year": [year],
                "month": [month],
                "day": days_for_month(year, month),
                "time": config.times,
                "area": config.area,
                "data_format": config.data_format,
                "download_format": config.download_format,
            }
            requests.append(
                DownloadRequest(
                    year=year,
                    month=month,
                    output_path=str(outfile),
                    payload=payload,
                )
            )

    return requests
