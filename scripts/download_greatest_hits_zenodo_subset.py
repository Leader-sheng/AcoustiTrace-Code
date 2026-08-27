"""Recover the 143 Greatest Hits I2AV-LAT references from Zenodo shards.

The original Greatest Hits ZIP is hosted on a legacy server that can be
unreachable from some networks.  SyncFusion publishes an uncompressed TAR
representation of the same dataset on Zenodo.  This script downloads those TAR
files into a resumable archive cache, keeps only the selected
``.resampled.wav`` files and conditioning frames (frame index 1), verifies the
released conditioning images, and writes the five-second evaluator references expected by
AcoustiTrace.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import requests
from tqdm import tqdm


ZENODO_RECORD = "12634671"
PRIMARY_SHARDS = (
    "train_shard_1.tar",
    "train_shard_2.tar",
    "train_shard_3.tar",
    "val_shard_1.tar",
    "test_onset_preds.tar",
)
FALLBACK_SHARDS = ("test_onset_augment_preds.tar",)


class ResumableHTTPReader:
    """Expose a reconnecting HTTP range request as one sequential byte stream."""

    def __init__(
        self,
        url: str,
        progress: tqdm[object],
        timeout: float,
        retries: int,
        retry_delay: float,
        start_position: int = 0,
    ) -> None:
        self.url = url
        self.progress = progress
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay
        self.position = start_position
        self.total: int | None = None
        self.session = requests.Session()
        self.response: requests.Response | None = None
        self.failures = 0
        self._connect_with_retry("initial connection")

    def _connect(self) -> None:
        headers = {"Range": f"bytes={self.position}-"} if self.position else {}
        response = self.session.get(
            self.url,
            headers=headers,
            stream=True,
            timeout=(self.timeout, self.timeout),
        )
        if response.status_code == 416 and self.position:
            complete = re.fullmatch(
                r"bytes \*/(\d+)", response.headers.get("content-range", "")
            )
            if complete and int(complete.group(1)) == self.position:
                self.total = self.position
                response.close()
                self.response = None
                return
        response.raise_for_status()
        if self.position and response.status_code != 206:
            response.close()
            raise RuntimeError(
                f"server ignored Range request at byte {self.position}: "
                f"HTTP {response.status_code}"
            )
        content_range = response.headers.get("content-range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range)
        if match:
            start = int(match.group(1))
            if start != self.position:
                response.close()
                raise RuntimeError(
                    f"server resumed at byte {start}, expected {self.position}"
                )
            if match.group(3) != "*":
                self.total = int(match.group(3))
        elif self.position:
            response.close()
            raise RuntimeError(f"missing Content-Range at byte {self.position}")
        else:
            length = int(response.headers.get("content-length", 0))
            self.total = length or None
        response.raw.decode_content = True
        self.response = response
        if self.total is not None and self.progress.total != self.total:
            self.progress.total = self.total
            self.progress.refresh()

    def _connect_with_retry(self, context: str) -> None:
        while True:
            try:
                self._connect()
                return
            except Exception as exc:
                if self.failures >= self.retries:
                    raise RuntimeError(
                        f"{context} failed at byte {self.position} after "
                        f"{self.failures} retries"
                    ) from exc
                self.failures += 1
                self.progress.write(
                    f"{context} failed at byte {self.position}; "
                    f"retry {self.failures}/{self.retries}: {exc}"
                )
                time.sleep(self.retry_delay)

    def _reconnect(self, error: BaseException) -> None:
        if self.failures >= self.retries:
            raise RuntimeError(
                f"download stopped at byte {self.position} after "
                f"{self.failures} reconnects"
            ) from error
        self.failures += 1
        if self.response is not None:
            self.response.close()
        self.progress.write(
            f"connection interrupted at byte {self.position}; "
            "reconnecting"
        )
        self._connect_with_retry("reconnect")

    def read(self, size: int = -1) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining != 0:
            if self.total is not None and self.position >= self.total:
                break
            request_size = 1024 * 1024 if remaining < 0 else remaining
            try:
                assert self.response is not None
                data = self.response.raw.read(request_size)
                if not data:
                    if self.total is None or self.position < self.total:
                        self._reconnect(EOFError("unexpected end of HTTP response"))
                        continue
                    break
            except Exception as exc:
                self._reconnect(exc)
                continue
            chunks.append(data)
            self.position += len(data)
            self.progress.update(len(data))
            if remaining > 0:
                remaining -= len(data)
        return b"".join(chunks)

    def close(self) -> None:
        if self.response is not None:
            self.response.close()
        self.session.close()


def read_source_map(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 143:
        raise ValueError(f"expected 143 source rows, found {len(rows)}")
    return rows


def archive_sample_id(filename: str, suffix: str) -> str | None:
    name = PurePosixPath(filename).name
    if not name.endswith(suffix):
        return None
    return name[: -len(suffix)] + "_mic"


def copy_member(archive: tarfile.TarFile, member: tarfile.TarInfo, target: Path) -> None:
    source = archive.extractfile(member)
    if source is None:
        raise RuntimeError(f"cannot read TAR member: {member.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    with source, temporary.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)
    temporary.replace(target)


def cached_samples(cache_dir: Path, wanted: set[str]) -> set[str]:
    return {
        sample_id
        for sample_id in wanted
        if (cache_dir / f"{sample_id}.wav").is_file()
        and (cache_dir / f"{sample_id}.jpg").is_file()
    }


def stream_shard(
    url: str,
    shard_name: str,
    wanted: set[str],
    cache_dir: Path,
    archive_cache_dir: Path,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> set[str]:
    """Download and scan one TAR shard, resuming its partial archive on restart."""

    completed_before = cached_samples(cache_dir, wanted)
    pending = wanted - completed_before
    if not pending:
        return set()
    have_audio = {sample_id for sample_id in pending if (cache_dir / f"{sample_id}.wav").is_file()}
    have_frame = {sample_id for sample_id in pending if (cache_dir / f"{sample_id}.jpg").is_file()}

    archive_cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_cache_dir / shard_name
    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    if not archive_path.is_file():
        downloaded = partial_path.stat().st_size if partial_path.is_file() else 0
        with tqdm(
            unit="B",
            unit_scale=True,
            desc=shard_name,
            initial=downloaded,
        ) as progress:
            reader = ResumableHTTPReader(
                url,
                progress,
                timeout,
                retries,
                retry_delay,
                start_position=downloaded,
            )
            try:
                mode = "ab" if downloaded else "wb"
                with partial_path.open(mode) as destination:
                    while True:
                        data = reader.read(1024 * 1024)
                        if not data:
                            break
                        destination.write(data)
                if reader.total is not None and reader.position != reader.total:
                    raise RuntimeError(
                        f"incomplete download for {shard_name}: "
                        f"{reader.position}/{reader.total} bytes"
                    )
            finally:
                reader.close()
        partial_path.replace(archive_path)

    scan_complete = False
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in tqdm(archive, desc=f"scan {shard_name}", unit="files"):
                if not member.isfile():
                    continue
                sample_id = archive_sample_id(member.name, ".resampled.wav")
                if sample_id in pending:
                    copy_member(archive, member, cache_dir / f"{sample_id}.wav")
                    have_audio.add(sample_id)
                else:
                    sample_id = archive_sample_id(member.name, ".frame_000001.jpg")
                    if sample_id in pending:
                        copy_member(archive, member, cache_dir / f"{sample_id}.jpg")
                        have_frame.add(sample_id)

                now_ready = have_audio & have_frame & pending
                pending -= now_ready
                if not pending:
                    break
        scan_complete = True
    finally:
        if scan_complete:
            archive_path.unlink(missing_ok=True)

    return cached_samples(cache_dir, wanted) - completed_before


def perceptual_hash(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("empty image")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(resized.astype(np.float32))[:8, :8]
    threshold = float(np.median(coefficients.reshape(-1)[1:]))
    return coefficients > threshold


def image_distance(conditioning_frame: Path, condition: Path) -> float:
    frame = cv2.imread(str(conditioning_frame), cv2.IMREAD_COLOR)
    reference = cv2.imread(str(condition), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"cannot read recovered conditioning frame: {conditioning_frame}")
    if reference is None:
        raise ValueError(f"cannot read conditioning image: {condition}")
    return float(np.mean(perceptual_hash(frame) != perceptual_hash(reference)))


def probe_duration(path: Path, ffprobe: str) -> float:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(result.stdout.strip())
    if not math.isfinite(duration):
        raise ValueError(f"invalid duration for {path}")
    return duration


def extract_audio(source: Path, target: Path, duration: float, ffmpeg: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part.wav")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-t",
            f"{duration:.6f}",
            "-ac",
            "1",
            "-ar",
            "22050",
            "-c:a",
            "pcm_s16le",
            str(temporary),
        ],
        check=True,
    )
    temporary.replace(target)


def build_references(
    root: Path,
    rows: list[dict[str, object]],
    cache_dir: Path,
    output_dir: Path,
    report_path: Path,
    max_distance: float,
    overwrite: bool,
    ffmpeg: str,
    ffprobe: str,
) -> int:
    report_rows: list[dict[str, object]] = []
    failures = 0
    for row in tqdm(rows, desc="I2AV-LAT references"):
        prompt_id = str(row["prompt_id"])
        sample_id = str(row["source_sample_id"])
        raw_audio = cache_dir / f"{sample_id}.wav"
        conditioning_frame = cache_dir / f"{sample_id}.jpg"
        target = output_dir / f"{prompt_id}.wav"
        report: dict[str, object] = {
            "prompt_id": prompt_id,
            "source_sample_id": sample_id,
            "source_audio_path": str(raw_audio),
            "source_duration_sec": "",
            "conditioning_frame_index": 1,
            "conditioning_frame_phash_distance": "",
            "reference_audio_path": str(target),
            "status": "failed",
            "error": "",
        }
        try:
            if not raw_audio.is_file() or not conditioning_frame.is_file():
                raise FileNotFoundError(f"source assets not recovered for {sample_id}")
            duration = probe_duration(raw_audio, ffprobe)
            report["source_duration_sec"] = duration
            clip_duration = float(row.get("target_clip_duration_sec", 5.0))
            if duration + 0.02 < clip_duration:
                raise ValueError(f"source is only {duration:.3f}s; need {clip_duration:.3f}s")
            condition = root / str(row["conditioning_asset_path"])
            if not condition.is_file():
                recovered = cv2.imread(str(conditioning_frame), cv2.IMREAD_COLOR)
                if recovered is None:
                    raise ValueError(
                        f"cannot read recovered conditioning frame: {conditioning_frame}"
                    )
                condition.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(condition), recovered):
                    raise ValueError(f"cannot write conditioning image: {condition}")
            distance = image_distance(conditioning_frame, condition)
            report["conditioning_frame_phash_distance"] = distance
            if distance > max_distance:
                raise ValueError(
                    "conditioning-frame mismatch: "
                    f"pHash distance {distance:.3f} exceeds {max_distance:.3f}"
                )
            if overwrite or not target.is_file():
                extract_audio(raw_audio, target, clip_duration, ffmpeg)
            report["status"] = "ready"
        except Exception as exc:
            failures += 1
            report["error"] = str(exc)
        report_rows.append(report)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)
    return failures


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-map",
        type=Path,
        default=root / "data" / "references" / "i2av_lat_sources.jsonl",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=root / "data" / "references" / "greatest_hits_zenodo",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "references" / "i2av_lat",
    )
    parser.add_argument(
        "--archive-cache-dir",
        type=Path,
        default=root / "data" / "references" / "greatest_hits_zenodo_archives",
        help="directory for resumable TAR downloads; completed shards are removed after scanning",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "outputs" / "i2av_lat_reference_report.csv",
    )
    parser.add_argument("--record", default=ZENODO_RECORD)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=50)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument(
        "--max-conditioning-frame-distance",
        "--max-first-frame-distance",
        dest="max_conditioning_frame_distance",
        type=float,
        default=0.25,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()

    rows = read_source_map(args.source_map)
    wanted = {str(row["source_sample_id"]) for row in rows}
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    ready = cached_samples(args.cache_dir, wanted)
    print(f"cached before download: {len(ready)}/{len(wanted)} source samples", flush=True)

    for shard in (*PRIMARY_SHARDS, *FALLBACK_SHARDS):
        if len(ready) == len(wanted):
            break
        # The direct file endpoint supports HTTP Range requests and is
        # substantially faster than the legacy ``?download=1`` redirect on
        # the PJLab development network.
        url = f"https://zenodo.org/records/{args.record}/files/{shard}"
        found = stream_shard(
            url,
            shard,
            wanted,
            args.cache_dir,
            args.archive_cache_dir,
            args.timeout,
            args.retries,
            args.retry_delay,
        )
        ready = cached_samples(args.cache_dir, wanted)
        print(
            f"{shard}: recovered {len(found)} new samples; total {len(ready)}/{len(wanted)}",
            flush=True,
        )

    missing = sorted(wanted - ready)
    if missing:
        print(f"warning: {len(missing)} source samples remain missing: {missing[:10]}", flush=True)

    failures = build_references(
        root,
        rows,
        args.cache_dir,
        args.output_dir,
        args.report,
        args.max_conditioning_frame_distance,
        args.overwrite,
        args.ffmpeg,
        args.ffprobe,
    )
    prepared = len(rows) - failures
    print(f"ready: {prepared}/143 reference audios; report: {args.report}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
