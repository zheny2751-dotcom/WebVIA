# -*- coding: utf-8 -*-
import os
import re
import json
from typing import Any, Dict, Iterable, Tuple, List

# ===== 手动填写：多个结果目录 & 输出路径 =====
RESULTS_DIRS: List[str] = [
    "verify_results/Verify_o4-mini-2025-04-16_html",
    "verify_results/Verify_claude-sonnet-4-20250514-thinking_html",
    "verify_results/Verify_gpt-5-2025-08-07_html",
    "verify_results/Verify_gemini-2.5-pro_html",
    "verify_results/Verify_claude-3-7-sonnet-20250219-thinking_html",
    "verify_results/Verify_gpt-4o-2024-11-20_html"
]

OUTPUT_JSON = "verify_pass_summary.json"  # 汇总输出路径（单个JSON文件）
WRITE_PER_FOLDER_DETAILS = True              # 是否为每个文件夹单独输出细节JSON

# ===== 纯色图过滤策略（在 bug 子文件夹中若发现纯色图，则跳过该 HTML 的 *_tasks.json）=====
EXCLUDE_BY_PURECOLOR = True      # 开关
PURECOLOR_TOLERANCE = 0          # 容差（0 表示必须完全相同；可设为 1~3 以容忍轻微压缩噪点）
IGNORE_ALPHA = True              # 检测纯色时是否忽略 alpha 通道

# ====== 依赖：Pillow（可选但强烈建议安装）=====
try:
    from PIL import Image
except Exception as _e:
    Image = None
    if EXCLUDE_BY_PURECOLOR:
        print("[WARN] 未安装 pillow，无法执行纯色图过滤，将忽略纯色图逻辑继续统计。")
        EXCLUDE_BY_PURECOLOR = False

# ====== 工具函数 ======

# 通用：从任意名字里提取“第一段数字”
_num_pat_anywhere = re.compile(r"(\d+)")

def extract_num_from_name(name: str) -> str:
    """从 '123__tasks.json' 或 'abc123.html' 中提取第一段编号 '123'."""
    m = _num_pat_anywhere.search(name)
    return m.group(1) if m else ""

def normalize_pass_val(v: Any) -> bool:
    """把多种表示方式规范为 bool 通过与否。"""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"pass", "passed", "true", "yes", "y", "ok", "success"}:
            return True
        if s in {"fail", "failed", "false", "no", "n", "x", "error"}:
            return False
    # 其他未知类型/空值一律当失败处理
    return False

def iter_task_passes(results_per_task: Any) -> Iterable[Tuple[str, bool]]:
    """
    兼容几种常见结构，产出 (task_name, pass_bool)：
    1) dict: { "taskA": {"pass": true}, "taskB": {"pass": "fail"}, ... }
    2) list: [ {"task": "taskA", "pass": true}, {"name":"taskB","pass":0}, ... ]
    3) dict: { "taskA": true, "taskB": "pass", ... }
    """
    if isinstance(results_per_task, dict):
        for k, v in results_per_task.items():
            if isinstance(v, dict):
                p = normalize_pass_val(v.get("pass"))
            else:
                p = normalize_pass_val(v)
            yield str(k), p
    elif isinstance(results_per_task, list):
        for item in results_per_task:
            if not isinstance(item, dict):
                continue
            name = str(item.get("task") or item.get("name") or item.get("id") or "")
            p = normalize_pass_val(item.get("pass"))
            yield name, p

def load_json_safe(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] 无法解析JSON: {path} ({e})")
        return {}

# ==== 纯色图检测 ====

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")

def _is_solid_color_image(img_path: str, tolerance: int = 0, ignore_alpha: bool = True) -> bool:
    """
    使用通道极值判断是否纯色图。若 RGB 通道各自 (max-min) <= tolerance，则判为纯色。
    """
    if Image is None:
        return False
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGBA")
            extrema = im.getextrema()  # [(min,max), ...] 长度 4
            channels = (0, 1, 2) if ignore_alpha else (0, 1, 2, 3)
            for ch in channels:
                mn, mx = extrema[ch]
                if (mx - mn) > tolerance:
                    return False
            return True
    except Exception as e:
        print(f"[WARN] 读图失败（忽略而不当纯色）: {img_path} ({e})")
        return False

def _bug_dir_for_results_dir(results_dir: str) -> str:
    """将 .../<parent>/Verify_xxx 映射到 .../<parent>/BUG_verify_xxx"""
    rd = results_dir.rstrip("/")
    parent = os.path.dirname(rd)
    base = os.path.basename(rd)
    if base.startswith("Verify"):
        bug_base = base.replace("Verify", "BUG_verify", 1)
    else:
        bug_base = "BUG_verify_" + base
    return os.path.join(parent, bug_base)

def _has_purecolor_in_bug_subdir(bug_base_dir: str, html_id: str) -> bool:
    """检测 bug 基础目录下名为 html_id 的子目录内是否存在任意纯色图。"""
    bug_subdir = os.path.join(bug_base_dir, str(html_id))
    if not os.path.isdir(bug_subdir):
        return False
    for root, _, files in os.walk(bug_subdir):
        for fn in files:
            if fn.lower().endswith(_IMAGE_EXTS):
                img_path = os.path.join(root, fn)
                if _is_solid_color_image(img_path, tolerance=PURECOLOR_TOLERANCE, ignore_alpha=IGNORE_ALPHA):
                    print(f"[BUG-PURE] 发现纯色图 → 跳过: {html_id} @ {img_path}")
                    return True
    return False

def summarize_one_dir(results_dir: str) -> Dict[str, Any]:
    """
    统计单个结果目录，返回结构化统计：
    {
      "dir": "...",
      "considered_files": X,
      "skipped_files": Y,
      "total_tasks": T,
      "passed_tasks": P,
      "pass_ratio": 0.1234,
      "details": [ { per-file 细节 ... }, ... ]
    }
    """
    stat = {
        "dir": results_dir,
        "considered_files": 0,
        "skipped_files": 0,
        "total_tasks": 0,
        "passed_tasks": 0,
        "pass_ratio": 0.0,
        "details": []
    }

    if not os.path.isdir(results_dir):
        print(f"[ERROR] 结果目录不存在: {results_dir}")
        return stat

    bug_base_dir = _bug_dir_for_results_dir(results_dir)
    if EXCLUDE_BY_PURECOLOR:
        print(f"[INFO] 纯色过滤: {results_dir} → BUG目录: {bug_base_dir}")
    else:
        print(f"[INFO] 不启用纯色过滤: {results_dir}")

    files = [fn for fn in os.listdir(results_dir) if fn.endswith("_tasks.json")]
    if not files:
        print(f"[INFO] 目录内无 *_tasks.json: {results_dir}")

    for fn in sorted(files):
        nid = extract_num_from_name(fn)
        full_path = os.path.join(results_dir, fn)

        # 纯色图规则：若 bug/<nid>/ 下存在任意纯色图，则跳过该 html 的统计
        if EXCLUDE_BY_PURECOLOR and nid:
            if _has_purecolor_in_bug_subdir(bug_base_dir, nid):
                stat["skipped_files"] += 1
                stat["details"].append({
                    "id": nid,
                    "file": fn,
                    "passed": 0,
                    "total": 0,
                    "ratio": None,
                    "skipped_because_purecolor": True
                })
                continue

        data = load_json_safe(full_path)
        rpt = data.get("results_per_task")
        if rpt is None:
            print(f"[INFO] 无 results_per_task: {fn}")
            stat["details"].append({
                "id": nid,
                "file": fn,
                "passed": 0,
                "total": 0,
                "ratio": None,
                "skipped_because_purecolor": False
            })
            continue

        file_total = 0
        file_pass = 0
        for task_name, p in iter_task_passes(rpt):
            file_total += 1
            if p:
                file_pass += 1

        stat["total_tasks"] += file_total
        stat["passed_tasks"] += file_pass
        stat["considered_files"] += 1

        ratio = (file_pass / file_total) if file_total else None
        print(f"[FILE] [{os.path.basename(results_dir)}] {fn}: {file_pass}/{file_total} tasks passed")
        stat["details"].append({
            "id": nid,
            "file": fn,
            "passed": file_pass,
            "total": file_total,
            "ratio": ratio,
            "skipped_because_purecolor": False
        })

    # 目录层面通过率
    if stat["total_tasks"] > 0:
        stat["pass_ratio"] = stat["passed_tasks"] / stat["total_tasks"]
    else:
        stat["pass_ratio"] = 0.0

    return stat

def write_folder_details_json(stat: Dict[str, Any]) -> None:
    """
    把某个目录的细节写成 <basename>_details.json
    结构：
    {
      "dir": "...",
      "considered_files": ...,
      "skipped_files": ...,
      "total_tasks": ...,
      "passed_tasks": ...,
      "pass_ratio": ...,
      "details": [ {id, file, passed, total, ratio, skipped_because_purecolor}, ... ]
    }
    """
    base = os.path.basename(stat["dir"].rstrip("/")) or stat["dir"]
    out_name = f"{base}_details.json"
    try:
        with open(out_name, "w", encoding="utf-8") as f:
            json.dump(stat, f, indent=2, ensure_ascii=False)
        print(f"[OK] 已写入细节JSON：{out_name}")
    except Exception as e:
        print(f"[ERROR] 写入细节JSON失败({out_name}): {e}")

# ====== 主流程：多目录统计并写入单一JSON + 每文件夹细节JSON ======
def main():
    all_stats = {}
    overall_total_tasks = 0
    overall_passed_tasks = 0
    overall_considered_files = 0
    overall_skipped_files = 0

    for rd in RESULTS_DIRS:
        stat = summarize_one_dir(rd)

        # 为每个结果目录写出细节 JSON
        if WRITE_PER_FOLDER_DETAILS:
            write_folder_details_json(stat)

        basekey = os.path.basename(rd.rstrip("/")) or rd
        all_stats[basekey] = {
            "dir": stat["dir"],
            "considered_files": stat["considered_files"],
            "skipped_files": stat["skipped_files"],
            "total_tasks": stat["total_tasks"],
            "passed_tasks": stat["passed_tasks"],
            "pass_ratio": stat["pass_ratio"],
            # 注意：总汇总中不再塞 details，避免文件过大；细节在各自 *_details.json
        }

        overall_total_tasks += stat["total_tasks"]
        overall_passed_tasks += stat["passed_tasks"]
        overall_considered_files += stat["considered_files"]
        overall_skipped_files += stat["skipped_files"]

    overall_ratio = (overall_passed_tasks / overall_total_tasks) if overall_total_tasks else 0.0

    summary = {
        "folders": all_stats,  # 每个文件夹的聚合统计
        "overall": {
            "considered_files": overall_considered_files,
            "skipped_files": overall_skipped_files,
            "total_tasks": overall_total_tasks,
            "passed_tasks": overall_passed_tasks,
            "pass_ratio": overall_ratio
        }
    }

    # 写入总汇总 JSON
    try:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("\n========== 汇总（总体） ==========")
        print(f"参与统计的结果文件数: {overall_considered_files}")
        print(f"被跳过(因 BUG 纯色图)的结果文件数: {overall_skipped_files}")
        print(f"通过任务数 / 任务总数: {overall_passed_tasks} / {overall_total_tasks}")
        print(f"总体 Pass 成功比例: {overall_ratio:.4f} ({overall_ratio*100:.2f}%)")
        print(f"\n[OK] 已写入汇总JSON: {OUTPUT_JSON}")
    except Exception as e:
        print(f"[ERROR] 写入输出JSON失败: {e}")

if __name__ == "__main__":
    main()
