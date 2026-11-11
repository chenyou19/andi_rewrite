"""Read-only cache census; write sizes and provenance to JSON and searchable HTML."""
from __future__ import annotations

from collections import Counter
import ctypes
from datetime import datetime, timedelta, timezone
import html
import json
import os
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/cache_inventory"


def normalized(value):
    return os.path.normcase(os.path.abspath(str(value)))


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def main():
    directories, candidates, reports, errors, links = {}, [], [], [], []
    class FileStandardInfo(ctypes.Structure):
        _fields_ = [("allocation_size", ctypes.c_longlong), ("end_of_file", ctypes.c_longlong),
                    ("number_of_links", ctypes.c_ulong), ("delete_pending", ctypes.c_ubyte),
                    ("directory", ctypes.c_ubyte)]

    kernel = ctypes.windll.kernel32
    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_ulong,
                                  ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.GetFileInformationByHandleEx.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                  ctypes.c_void_p, ctypes.c_ulong]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    allocation_failures = 0

    def allocated_size(path, fallback):
        nonlocal allocation_failures
        handle = kernel.CreateFileW(str(path), 0x80, 7, None, 3, 0, None)
        if handle == ctypes.c_void_p(-1).value:
            allocation_failures += 1
            return fallback
        try:
            info = FileStandardInfo()
            if not kernel.GetFileInformationByHandleEx(handle, 1, ctypes.byref(info), ctypes.sizeof(info)):
                allocation_failures += 1
                return fallback
            return info.allocation_size
        finally:
            kernel.CloseHandle(handle)
    for base, dirs, files in os.walk(ROOT, followlinks=False,
                                     onerror=lambda e: errors.append(str(e))):
        directory = Path(base)
        keep = []
        for name in dirs:
            child = directory / name
            if name in (".git", "node_modules") or child == OUT:
                continue
            try:
                if child.lstat().st_file_attributes & 0x400:
                    links.append(str(child))
                    continue
            except OSError as exc:
                errors.append(str(exc))
                continue
            keep.append(name)
        dirs[:] = keep
        direct = {"bytes": 0, "allocated_bytes": 0, "files": 0,
                  "extensions": Counter(), "earliest": None, "latest": None}
        for name in files:
            path = directory / name
            try:
                stat = path.stat()
                allocated = allocated_size(path, stat.st_size)
                direct["bytes"] += stat.st_size
                direct["allocated_bytes"] += allocated
                direct["files"] += 1
                direct["extensions"][path.suffix.lower() or "[no extension]"] += 1
                direct["earliest"] = min(direct["earliest"] or stat.st_mtime, stat.st_mtime)
                direct["latest"] = max(direct["latest"] or stat.st_mtime, stat.st_mtime)
                if name == "inference_report.json":
                    reports.append(path)
            except OSError as exc:
                errors.append(str(exc))
        directories[directory] = direct
        if "cache" in directory.name.lower() or directory.name == "reference_cases":
            candidates.append(directory)
    # Sum subtrees once, bottom-up. Cache records use disjoint roots below.
    totals = {}
    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        own = directories[directory]
        total = {**own, "extensions": own["extensions"].copy()}
        child_values = [totals[p] for p in directory.iterdir()
                        if p in totals and p.parent == directory]
        for child in child_values:
            for key in ("bytes", "allocated_bytes", "files"):
                total[key] += child[key]
            total["extensions"].update(child["extensions"])
            if child["earliest"] is not None:
                total["earliest"] = min(total["earliest"] or child["earliest"], child["earliest"])
                total["latest"] = max(total["latest"] or child["latest"], child["latest"])
        totals[directory] = total
    roots = [p for p in candidates if not any(a in candidates for a in p.parents)]
    # A name ending in "cached" may describe an experiment, not a data cache.
    # Keep model/report output directories out of cache-only cleanup inventories.
    excluded_experiment_outputs = []
    for directory in list(roots):
        extensions = totals[directory]["extensions"]
        if (re.search(r"cached\d*$", directory.name, re.IGNORECASE)
                and not (extensions[".npz"] or extensions[".npy"])
                and (extensions[".pt"] or extensions[".py"])):
            excluded_experiment_outputs.append(str(directory))
            roots.remove(directory)
    # These named roots are containers for distinct experiments/input pools.
    container_roots = {ROOT / "outputs/eval_cache",
                       ROOT / "outputs/diagnostics/domain_classifier/cache"}
    container_roots.update(p for p in roots if p.name == "brats_cache")
    expanded = []
    for directory in roots:
        if directory in container_roots:
            expanded.extend(p for p in directories if p.parent == directory)
            if directories[directory]["files"]:
                expanded.append(directory)
        else:
            expanded.append(directory)
    report_map = {}
    for path in reports:
        report = load_json(path)
        if not report:
            continue
        value = report.get("evaluation_result", {}).get("cache_directory")
        if not value:
            value = report.get("full_config_snapshot", {}).get("evaluation", {}).get("cache", {}).get("directory")
        if value:
            report_map.setdefault(normalized(value), []).append((path, report))
    records = []
    zone = timezone(timedelta(hours=8))
    for directory in expanded:
        size = directories[directory] if directory in container_roots else totals[directory]
        relative = directory.relative_to(ROOT).as_posix()
        manifest = load_json(directory / "manifest.json")
        entries = manifest.get("entries", []) if isinstance(manifest, dict) else []
        evaluation_cache = (isinstance(manifest, dict) and "raw_fingerprint" in manifest
                            and isinstance(entries, list))
        if directory.name == "__pycache__" or directory.name == ".pytest_cache":
            category = "tool_cache"
        elif "test_tmp" in relative or "/attempt_" in relative:
            category = "diagnostic_test_or_attempt_cache"
        elif evaluation_cache:
            category = "inference_cache"
        else:
            category = "preprocessing_or_analysis_cache"
        sources = []
        pairs = report_map.get(normalized(directory), [])
        fallback = directory.parent / "inference_report.json"
        if not pairs and fallback.exists():
            data = load_json(fallback)
            if data:
                pairs = [(fallback, data)]
        for path, report in pairs:
            settings = report.get("inference_settings", {})
            basic = report.get("basic_info", {})
            checkpoint = settings.get("checkpoint_path")
            sources.append({"report": str(path), "experiment": basic.get("experiment_name"),
                            "checkpoint": checkpoint, "checkpoint_name": Path(checkpoint).name if checkpoint else None,
                            "subjects": settings.get("number_of_cases"),
                            "csv": settings.get("scan_csv_or_test_csv_path"),
                            "start_time": basic.get("start_time"), "end_time": basic.get("end_time")})
        if "outputs/runs/" in relative:
            run_name = relative.split("outputs/runs/", 1)[1].split("/", 1)[0]
            owner = run_name
        elif "mixed_source_no_iqr_mean888" in relative:
            owner = "來源端無 IQR 比較／raw＋matching LMDB 建置，FOMO 原始影像重用快取"
        elif directory.name == "reference_cases":
            owner = "BraTS 健康組織參考 quantiles；histogram matching 資料建置"
        elif "domain_classifier" in relative:
            owner = "domain classifier：模型輸入建置／分布分析／測試；依路徑細分"
        elif sources:
            owner = sources[0].get("experiment") or directory.name
        elif category == "tool_cache":
            owner = "Python bytecode／pytest 工具快取，無單一訓練 checkpoint"
        else:
            owner = "未找到可唯一對應的推論報告；保留路徑作追溯"
        record = {"path": str(directory), "relative_path": relative, "category": category,
                  "owner": owner, "bytes": size["bytes"], "allocated_bytes": size["allocated_bytes"],
                  "gib": size["bytes"] / 1024**3,
                  "allocated_gib": size["allocated_bytes"] / 1024**3,
                  "file_count": size["files"], "extensions": dict(size["extensions"]),
                  "first_file_mtime": datetime.fromtimestamp(size["earliest"], zone).isoformat() if size["earliest"] else None,
                  "last_file_mtime": datetime.fromtimestamp(size["latest"], zone).isoformat() if size["latest"] else None,
                  "inference_sources": sources, "manifest": str(directory / "manifest.json") if manifest else None}
        if evaluation_cache:
            record.update(subjects_cached=len(entries), collection_complete=manifest.get("collection_complete"),
                          mf_subjects=sum("mf" in e for e in entries),
                          labeled_subjects=sum(bool(e.get("has_label")) for e in entries))
        if category == "inference_cache" and not sources:
            record["attribution_status"] = "manifest_and_path_only_no_report"
        else:
            record["attribution_status"] = "report" if sources else "path_and_producer_code"
        records.append(record)
    records.sort(key=lambda row: row["bytes"], reverse=True)
    groups = {}
    for row in records:
        key = row["category"]
        group = groups.setdefault(key, {"locations": 0, "bytes": 0, "allocated_bytes": 0, "files": 0})
        group["locations"] += 1
        group["bytes"] += row["bytes"]
        group["allocated_bytes"] += row["allocated_bytes"]
        group["files"] += row["file_count"]
    for value in groups.values():
        value["gib"] = value["bytes"] / 1024**3
        value["allocated_gib"] = value["allocated_bytes"] / 1024**3
    report = {"scanned_at": datetime.now(zone).isoformat(), "workspace": str(ROOT),
              "units": "GiB = 1024^3 bytes", "cache_locations": len(records),
              "bytes": sum(r["bytes"] for r in records),
              "allocated_bytes": sum(r["allocated_bytes"] for r in records),
              "scope": "Workspace cache-named roots, reference_cases, and separate child pools of cache containers; "
                       "excludes published LMDB, checkpoints, prediction exports, external data and external environments.",
              "size_notes": "Windows FileStandardInfo.AllocationSize measures allocated bytes. Hardlink aliases, if any, "
                            "are counted per path; these totals are not a promised reclaimable-space amount.",
              "allocation_measurement_failures": allocation_failures,
              "excluded_cache_named_experiment_outputs": excluded_experiment_outputs,
              "excluded_directory_names": [".git", "node_modules"], "skipped_reparse_points": links,
              "scan_errors": errors, "groups": groups, "records": records,
              "workspace_volume_free_gib": shutil.disk_usage(ROOT).free / 1024**3}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inventory.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    headings = ["分類", "來源實驗／用途", "Cache 路徑", "GiB", "磁碟配置 GiB", "檔案數", "個案數", "Checkpoint／推論時間"]
    rows = []
    for row in records:
        source_text = "; ".join(f"{s.get('checkpoint_name') or '?'} | {s.get('start_time') or '?'} | {s.get('report')}"
                                for s in row["inference_sources"])
        cells = [row["category"], row["owner"], row["path"], f"{row['gib']:.6f}",
                 f"{row['allocated_gib']:.6f}", str(row["file_count"]),
                 str(row.get("subjects_cached", "—")), source_text or row["attribution_status"]]
        rows.append("<tr>" + "".join("<td>" + html.escape(value) + "</td>" for value in cells) + "</tr>")
    page = """<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>Workspace cache inventory</title>
<style>body{font-family:Segoe UI,Microsoft JhengHei,sans-serif;margin:24px;color:#182433}h1{font-size:24px}p{max-width:1000px;line-height:1.6}input{padding:10px;width:550px;max-width:90%;margin-bottom:16px}table{border-collapse:collapse;width:100%;font-size:13px}th{background:#e9eff7;position:sticky;top:0;cursor:pointer}td,th{padding:9px;border:1px solid #dbe2ea;text-align:left;vertical-align:top}td:nth-child(3),td:nth-child(8){overflow-wrap:anywhere;min-width:240px}td:nth-child(4),td:nth-child(5),td:nth-child(6),td:nth-child(7){text-align:right}tr:nth-child(even){background:#f8fafc}</style>
<h1>工作區 Cache 盤點</h1>"""
    page += f"<p>{html.escape(report['scanned_at'])}<br>{html.escape(str(ROOT))}<br>共 {len(records)} 個互不重疊的 cache 位置，檔案大小 {report['bytes']/1024**3:.3f} GiB，磁碟配置約 {report['allocated_bytes']/1024**3:.3f} GiB。全部保留。</p>"
    page += "<p>GiB＝1024³ bytes。列出具名 cache、reference_cases 與 cache 容器內的各實驗。LMDB、checkpoint、prediction NIfTI 與外部資料不計入。分類來源依 manifest、推論報告或 producer 程式／路徑；工具 cache 無單一訓練歸屬。磁碟配置以 Windows API 計算，若有 hardlink 路徑則會按路徑計數。</p>"
    page += '<input id="search" placeholder="搜尋實驗、checkpoint、cache 路徑…"><table><thead><tr>'
    page += "".join(f'<th onclick="sortRows({i})">{html.escape(label)}</th>' for i, label in enumerate(headings))
    page += "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    page += """<script>const body=document.querySelector('tbody');document.querySelector('#search').oninput=e=>{const q=e.target.value.toLowerCase();for(const r of body.rows)r.hidden=!r.textContent.toLowerCase().includes(q)};let order=-1;function sortRows(n){order*=-1;const rs=[...body.rows];rs.sort((a,b)=>{const x=a.cells[n].textContent,y=b.cells[n].textContent;return order*((n>=3&&n<=6)?(Number(x)||0)-(Number(y)||0):x.localeCompare(y))});rs.forEach(r=>body.append(r))}</script></html>"""
    (OUT / "inventory.html").write_text(page, encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=True))


if __name__ == "__main__":
    main()
