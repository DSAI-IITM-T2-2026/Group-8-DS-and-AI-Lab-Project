"""CDS downloads with multi-account parallelism, retry, and progress tracking."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cdsapi
from tqdm import tqdm

from pipeline.checkpoint import CheckpointManager
from pipeline.config import PipelineConfig
from pipeline.credentials import CdsAccount, create_client, load_accounts
from pipeline.request_builder import DownloadRequest, build_monthly_requests
from pipeline.work_partition import partition_requests

logger = logging.getLogger(__name__)

QUEUE_LIMIT_HINT = (
    "CDS queue is full for account '{account}'. Waiting for queued jobs to clear. "
    "Check/cancel stale jobs at https://cds.climate.copernicus.eu"
)
LICENCE_HINT = (
    "Account '{account}' has not accepted the ERA5 licence. "
    "Log in as that CDS user and accept it at: "
    "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download#manage-licences"
)


def _is_queue_limit_error(error: str) -> bool:
    lowered = error.lower()
    return "queued requests" in lowered or "temporarily limited" in lowered


def _is_licence_error(error: str) -> bool:
    lowered = error.lower()
    return "licence" in lowered or "license" in lowered


def _download_one(
    client: cdsapi.Client,
    request: DownloadRequest,
    config: PipelineConfig,
    checkpoint: CheckpointManager,
    log_path: Path,
    account_name: str,
) -> tuple[str, bool, str | None]:
    outfile = Path(request.output_path)
    key = request.key

    if outfile.exists() and outfile.stat().st_size > 0:
        checkpoint.mark_completed(key, account=account_name)
        return key, True, None

    if checkpoint.is_completed(key):
        return key, True, None

    delay = config.retry_base_delay
    last_error: str | None = None
    attempt = 0
    queue_attempts = 0

    while attempt < config.max_retries:
        attempt += 1
        try:
            outfile.parent.mkdir(parents=True, exist_ok=True)
            client.retrieve(config.dataset, request.payload, str(outfile))

            if not outfile.exists() or outfile.stat().st_size == 0:
                raise RuntimeError(f"Download produced empty file: {outfile}")

            checkpoint.mark_completed(key, account=account_name)
            with open(log_path, "a") as log:
                log.write(f"SUCCESS {key} account={account_name} attempt={attempt}\n")
            return key, True, None

        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "[%s] Download %s attempt %d failed: %s",
                account_name,
                key,
                attempt,
                exc,
            )
            with open(log_path, "a") as log:
                log.write(f"FAIL {key} account={account_name} attempt={attempt} error={exc}\n")

            if outfile.exists():
                outfile.unlink(missing_ok=True)

            if _is_licence_error(last_error):
                logger.error(LICENCE_HINT.format(account=account_name))
                break

            if _is_queue_limit_error(last_error):
                queue_attempts += 1
                if queue_attempts > config.queue_limit_max_retries:
                    break
                logger.warning(QUEUE_LIMIT_HINT.format(account=account_name))
                logger.info(
                    "[%s] Queue full — retrying %s in %d seconds (queue attempt %d/%d)...",
                    account_name,
                    key,
                    config.queue_limit_delay,
                    queue_attempts,
                    config.queue_limit_max_retries,
                )
                time.sleep(config.queue_limit_delay)
                attempt -= 1
                continue

            if attempt < config.max_retries:
                logger.info("[%s] Retrying %s in %d seconds...", account_name, key, delay)
                time.sleep(delay)
                delay *= 2

    checkpoint.mark_failed(key, last_error or "Unknown error", account=account_name)
    return key, False, last_error


def _account_worker(
    account: CdsAccount,
    client: cdsapi.Client,
    requests: list[DownloadRequest],
    config: PipelineConfig,
    checkpoint: CheckpointManager,
    log_path: Path,
    startup_offset: int,
    progress: tqdm,
    stats: dict[str, int],
    stats_lock: threading.Lock,
) -> None:
    if startup_offset > 0:
        logger.info("[%s] Staggered startup — waiting %d seconds", account.name, startup_offset)
        time.sleep(startup_offset)

    if config.startup_delay_seconds > 0 and startup_offset == 0:
        logger.info(
            "[%s] Startup cooldown — waiting %d seconds",
            account.name,
            config.startup_delay_seconds,
        )
        time.sleep(config.startup_delay_seconds)

    logger.info("[%s] Processing %d assigned downloads", account.name, len(requests))

    for req in requests:
        key, ok, _ = _download_one(client, req, config, checkpoint, log_path, account.name)
        with stats_lock:
            if ok:
                stats["success"] += 1
                if config.request_delay_seconds > 0:
                    time.sleep(config.request_delay_seconds)
            else:
                stats["failed"] += 1
        progress.set_postfix({"last": key, "account": account.name, "ok": ok})
        progress.update(1)


def run_downloads(config: PipelineConfig) -> dict[str, int]:
    accounts = load_accounts(config.cds_accounts)
    requests = build_monthly_requests(config)
    checkpoint = CheckpointManager(config.checkpoint_file())
    log_path = config.download_log()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    pending = [
        req for req in requests
        if not checkpoint.is_completed(req.key) and not Path(req.output_path).exists()
    ]

    partitions = partition_requests(pending, accounts, config.partition_strategy)

    logger.info(
        "Downloads: %d total, %d completed, %d pending across %d CDS account(s)",
        len(requests),
        checkpoint.completed_count,
        len(pending),
        len(accounts),
    )
    for account in accounts:
        logger.info("  [%s] → %d files", account.name, len(partitions[account.name]))

    if not pending:
        return {"total": len(requests), "success": len(requests), "failed": 0}

    clients = {account.name: create_client(account) for account in accounts}
    stats = {"success": checkpoint.completed_count, "failed": 0}
    stats_lock = threading.Lock()

    with tqdm(total=len(pending), desc="Downloading ERA5", unit="file") as progress:
        with ThreadPoolExecutor(max_workers=len(accounts)) as executor:
            futures = [
                executor.submit(
                    _account_worker,
                    account,
                    clients[account.name],
                    partitions[account.name],
                    config,
                    checkpoint,
                    log_path,
                    index * config.account_stagger_seconds,
                    progress,
                    stats,
                    stats_lock,
                )
                for index, account in enumerate(accounts)
            ]
            for future in as_completed(futures):
                future.result()

    return {
        "total": len(requests),
        "success": stats["success"],
        "failed": stats["failed"],
    }
