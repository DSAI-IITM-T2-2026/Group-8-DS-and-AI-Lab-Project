"""Partition download requests across multiple CDS accounts."""

from __future__ import annotations

from pipeline.credentials import CdsAccount
from pipeline.request_builder import DownloadRequest


def partition_requests(
    requests: list[DownloadRequest],
    accounts: list[CdsAccount],
    strategy: str = "round_robin",
) -> dict[str, list[DownloadRequest]]:
    buckets: dict[str, list[DownloadRequest]] = {account.name: [] for account in accounts}

    if not accounts:
        return buckets

    for index, request in enumerate(requests):
        if strategy == "year_modulo":
            account = accounts[int(request.year) % len(accounts)]
        else:
            account = accounts[index % len(accounts)]
        buckets[account.name].append(request)

    return buckets
