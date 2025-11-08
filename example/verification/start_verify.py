# -*- coding: utf-8 -*-
import requests
import base64
import time
import json
import subprocess
import psutil
import os
import hashlib
import sys
import random
from tqdm import tqdm
import concurrent.futures
from multiprocessing import Process, Queue, cpu_count
import re
import openai
from openai import OpenAI
import math
import ast  # ★ find_node_by_id_for_prompt 用到 ast
from collections import defaultdict
import shutil
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from queue import Empty as QueueEmpty
from datetime import datetime
import argparse
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

# =========================================================
# ===================== 配置对象 ===========================
# =========================================================
@dataclass(frozen=True)
class RunnerConfig:
    # 必填（与你原脚本 1:1 对齐）
    input_type: str                 # "agent" | "manual"
    output_image_dir: str           # 原 IMAGE_DIR
    input_dir: str                  # HTML 目录
    task_source_jsonl: str          # JSONL（id -> tasks）
    model_name: str                 # 调用的模型名
    bug_dir: str                    # BUG_DIR
    webenv_path: str                # WEBENV 脚本路径

    # 可选
    num_port: int = 20              # 并发端口数
    port_base: int = 8000           # 端口起点
    api_base: str = ""              # OpenAI 兼容 BASE
    api_key: str = ""               # OpenAI KEY

    # 超时与令牌（与脚本默认一致）
    default_timeout: int = 30       # 本地 HTTP 超时(s)
    model_timeout: int = 1000       # LLM 单次超时(s)
    max_completion_tokens: int = 10000

# =========================================================
# ===================== 配置加载 ===========================
# =========================================================
def load_config_or_exit() -> Tuple[RunnerConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="WEBVIA multi-task explorer (dataclass-config)")
    parser.add_argument("--config", required=True, help="path to config.json")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] config file not found: {args.config}")
        raise SystemExit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 必填校验
    required = [
        "input_type", "output_image_dir", "input_dir",
        "task_source_jsonl", "model_name", "bug_dir", "webenv_path"
    ]
    missing = [k for k in required if k not in cfg or cfg[k] in (None, "")]
    if missing:
        print(f"[ERROR] missing required fields in config: {missing}")
        raise SystemExit(1)

    rc = RunnerConfig(
        input_type=str(cfg["input_type"]).strip().lower(),
        output_image_dir=cfg["output_image_dir"],
        input_dir=cfg["input_dir"],
        task_source_jsonl=cfg["task_source_jsonl"],
        model_name=cfg["model_name"],
        bug_dir=cfg["bug_dir"],
        webenv_path=cfg["webenv_path"],
        num_port=int(cfg.get("num_port", 20)),
        port_base=int(cfg.get("port_base", 8000)),
        api_base=cfg.get("api_base", "") or "",
        api_key=cfg.get("api_key", "") or "",
        default_timeout=int(cfg.get("default_timeout", 30)),
        model_timeout=int(cfg.get("model_timeout", 1000)),
        max_completion_tokens=int(cfg.get("max_completion_tokens", 10000)),
    )

    # 额外基本检查
    if rc.input_type not in ("agent", "manual"):
        print(f"[ERROR] input_type must be 'agent' or 'manual', got={rc.input_type}")
        raise SystemExit(1)

    return rc, args

# =========================================================
# ============ 兼容层：把旧的全局名映射回来 =================
# =========================================================
def _bind_config_aliases(CONFIG: RunnerConfig, CLIENT: OpenAI):
    """
    为尽量少改你原函数，恢复你脚本里用到的全局名字（只映射配置/客户端相关）。
    后续你也可以逐步把函数签名改成传 rc/client，而不再依赖这些别名。
    """
    g = globals()
    # OpenAI / client
    g["client"] = CLIENT
    openai.api_base = CONFIG.api_base
    openai.api_key  = CONFIG.api_key

    # 旧名别名（只读常量/配置）
    g["INPUT_TYPE"]          = CONFIG.input_type
    g["IMAGE_DIR"]           = CONFIG.output_image_dir
    g["input_dir"]           = CONFIG.input_dir
    g["TASKS_SOURCE_JSONL"]  = CONFIG.task_source_jsonl
    g["MODEL_NAME"]          = CONFIG.model_name
    g["BUG_DIR"]             = CONFIG.bug_dir
    g["WEBENV"]              = CONFIG.webenv_path
    g["NUM_PORT"]            = CONFIG.num_port
    g["PORT_BASE"]           = CONFIG.port_base
    g["DEFAULT_TIMEOUT"]     = CONFIG.default_timeout
    g["MODEL_TIMEOUT"]       = CONFIG.model_timeout
    g["MODEL_MAX_TOKENS"]    = CONFIG.max_completion_tokens

# =========================================================
# ================= 你原有的工具/逻辑函数 ==================
# =========================================================


# ===== 显式声明所有会在函数中使用的“全局名”并给默认值（spawn 安全） =====
client: Optional[OpenAI] = None
SESSION = requests.Session(); SESSION.trust_env = False

# 配置别名（由 _bind_config_aliases 注入正确值；这里给默认占位）
INPUT_TYPE: str = "agent"
IMAGE_DIR: str = "./images"
input_dir: str = "./htmls"
TASKS_SOURCE_JSONL: str = "./input_data/data.jsonl"
MODEL_NAME: str = "gpt-5"
BUG_DIR: str = "./buglogs"
WEBENV: str = "./webenv.py"
NUM_PORT: int = 1
PORT_BASE: int = 8000
DEFAULT_TIMEOUT: int = 30
MODEL_TIMEOUT: int = 1000
MODEL_MAX_TOKENS: int = 10000

def http_get(url, timeout=None, **kw):
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    r = SESSION.get(url, timeout=timeout, **kw)
    r.raise_for_status()
    return r

def http_post(url, timeout=None, **kw):
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    r = SESSION.post(url, timeout=timeout, **kw)
    r.raise_for_status()
    return r

def _call_llm(messages, max_tokens):
    # 使用映射回来的 MODEL_NAME / client
    return client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=max_tokens,
    )

def safe_llm_call(messages, max_tokens=None, timeout=None):
    """线程池套一层超时，防止底层 SDK 卡住。"""
    if max_tokens is None:
        max_tokens = MODEL_MAX_TOKENS
    if timeout is None:
        timeout = MODEL_TIMEOUT
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_call_llm, messages, max_tokens)
        try:
            resp = fut.result(timeout=timeout)
            return (resp.choices[0].message.content or "").strip()
        except FuturesTimeout:
            raise TimeoutError(f"LLM call exceeded {timeout}s")

# ===== 其余你的函数体保留不变 =====
# （从这里开始，贴回你原脚本的所有函数：_safe_write_text/_safe_write_json/.../multi_task_explore_html/.../worker 等。
#  唯一注意：这些函数里引用的 IMAGE_DIR/BUG_DIR/MODEL_NAME/MODEL_TIMEOUT/... 都已经由 _bind_config_aliases() 注入完毕。）

# -------------------- 工具函数：tasks 生成 --------------------
ACTION_MARKERS = {"input", "click", "select"}
BUTTON_WORDS = {
    "添加成就", "删除", "编辑", "保存成就", "保存", "确认", "提交", "关闭", "取消"
}

def _safe_write_text(path, text):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("" if text is None else str(text))
    except Exception as e:
        print(f"[BUG-LOG] write text failed {path}: {e}")

def _safe_write_json(path, obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[BUG-LOG] write json failed {path}: {e}")

def _safe_save_b64_png(b64_data, path):
    try:
        if b64_data:
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64_data))
    except Exception as e:
        print(f"[BUG-LOG] save image failed {path}: {e}")

def _ensure_original_html(html_src_path: str, html_dir: Path, ext: str):
    try:
        dst = html_dir / f"original{ext}"
        if not dst.exists():
            shutil.copy(html_src_path, dst)
    except Exception as e:
        print(f"[BUG-LOG] copy html failed: {e}")

_idx_pat = re.compile(r"_(\d{3})\.")

def _next_issue_index(html_dir: Path) -> int:
    max_idx = -1
    if not html_dir.exists():
        return 0
    for p in html_dir.iterdir():
        m = _idx_pat.search(p.name)
        if m:
            try:
                i = int(m.group(1))
                if i > max_idx:
                    max_idx = i
            except ValueError:
                pass
    return max_idx + 1

def cp_bad_html(
    src_file,
    *,
    reason: str = "",
    task: str | None = None,
    step: dict | None = None,
    actions: list | None = None,
    responses: list | None = None,
    domtree: dict | str | None = None,
    img_b64: str | None = None,
    model_output: str | None = None,
    inputs: dict | None = None,
    exception: Exception | str | None = None,
    extra: dict | None = None
):
    basename = os.path.basename(src_file)
    stem, ext = os.path.splitext(basename)
    html_dir = Path(BUG_DIR) / stem
    html_dir.mkdir(parents=True, exist_ok=True)
    _ensure_original_html(src_file, html_dir, ext)

    idx = _next_issue_index(html_dir)
    idx3 = f"{idx:03d}"
    print(f"正在记录 BUG：{src_file} | 原因={reason or 'unspecified'} | 索引={idx3}")

    record = {
        "index": idx,
        "reason": reason,
        "html_file": os.path.abspath(src_file),
        "task": task,
        "step": step,
        "actions": actions,
        "responses": responses,
        "inputs": inputs,
        "extra": extra,
    }
    _safe_write_json(html_dir / f"bug_{idx3}.json", record)
    try:
        with open(html_dir / "bug_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[BUG-LOG] append bug_log.jsonl failed: {e}")

    if domtree is not None:
        if isinstance(domtree, (dict, list)):
            _safe_write_json(html_dir / f"domtree_{idx3}.json", domtree)
        else:
            _safe_write_text(html_dir / f"domtree_{idx3}.txt", domtree)

    if img_b64:
        _safe_save_b64_png(img_b64, html_dir / f"input_{idx3}.png")

    if model_output:
        _safe_write_text(html_dir / f"model_output_{idx3}.txt", model_output)

    if exception:
        if isinstance(exception, Exception):
            tb = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
            _safe_write_text(html_dir / f"exception_{idx3}.txt", tb)
        else:
            _safe_write_text(html_dir / f"exception_{idx3}.txt", str(exception))

def _basename_stem(p: str) -> str:
    name = os.path.basename(p)
    if name.endswith(".png"):
        name = name[:-4]
    return name

def _collapse_underscores(s: str) -> str:
    return re.sub(r"_+", "_", s)

def _parse_stem_to_task(stem: str) -> str:
    if stem == "start":
        return ""
    s = stem.lstrip("_")
    s = _collapse_underscores(s)
    tokens = [t for t in s.split("_") if t != ""]
    parts = []
    i = 0
    ACTION_MARKERS = {"input", "click", "select"}
    BUTTON_WORDS = {"添加成就", "删除", "编辑", "保存成就", "保存", "确认", "提交", "关闭", "取消"}
    while i < len(tokens):
        tok = tokens[i]
        if tok == "input":
            i += 1
            if i < len(tokens) and re.fullmatch(r"\d+", tokens[i] or ""):
                i += 1
            val_tokens = []
            while i < len(tokens):
                t = tokens[i]
                if (t in ACTION_MARKERS) or (t in BUTTON_WORDS):
                    break
                val_tokens.append(t)
                i += 1
            value = _collapse_underscores("_".join(val_tokens)).strip("_")
            parts.append(f"input{value}" if value else "input")
            continue
        if tok in ("click", "select"):
            i += 1
            if i < len(tokens) and re.fullmatch(r"\d+", tokens[i] or ""):
                i += 1
            label_tokens = []
            while i < len(tokens):
                t = tokens[i]
                if (t in ACTION_MARKERS) or (t in BUTTON_WORDS):
                    break
                label_tokens.append(t)
                i += 1
            label = _collapse_underscores("_".join(label_tokens)).strip("_")
            parts.append(label if label else tok)
            continue
        if tok in BUTTON_WORDS:
            parts.append("—保存成功" if tok == "保存成就" else tok)
            i += 1
            continue
        parts.append(tok)
        i += 1
    task = "-".join([p for p in parts if p]).strip("-").strip()
    if task.startswith("start-"):
        task = task[6:]
    return task

def _tasks_from_operation_info(op_info):
    tasks = []
    for op in op_info:
        end_path = op.get("end_image_path") or ""
        stem = _basename_stem(end_path)
        task = _parse_stem_to_task(stem)
        if task:
            tasks.append(task)
    seen = set(); uniq = []
    for t in tasks:
        if t not in seen:
            uniq.append(t); seen.add(t)
    return uniq

def _tasks_from_image_list(image_list):
    if not image_list or len(image_list) < 2:
        return []
    stems = [_basename_stem(p) for p in image_list]
    stems = [s for s in stems if s and s != "start"]
    save_like = [s for s in stems if s.endswith("_保存成就")]
    save_set = set(save_like)

    chosen = set()
    for s in stems:
        longer_exists = any((t != s and t.startswith(s)) for t in stems)
        if not longer_exists:
            chosen.add(s)
    chosen |= save_set

    tasks = []
    for s in stems:
        if s in chosen:
            task = _parse_stem_to_task(s)
            if task:
                tasks.append(task)
    seen = set(); uniq = []
    for t in tasks:
        if t not in seen:
            uniq.append(t); seen.add(t)
    return uniq

def load_id2tasks_from_jsonl(jsonl_path: str, mode: str):
    id2tasks = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = str(rec.get("id") or rec.get("html_id") or rec.get("name") or "").strip()
            if not rid:
                continue
            if mode == "manual":
                tasks_field = rec.get("tasks", [])
                if isinstance(tasks_field, list):
                    seen = set()
                    tasks = []
                    for t in tasks_field:
                        if isinstance(t, str):
                            t2 = t.strip()
                            if t2 and t2 not in seen:
                                tasks.append(t2); seen.add(t2)
                    id2tasks[rid] = tasks
                else:
                    id2tasks[rid] = []
            else:
                op_info = rec.get("operation_info") or []
                image_list = rec.get("image_list") or []
                if op_info:
                    tasks = _tasks_from_operation_info(op_info)
                else:
                    tasks = _tasks_from_image_list(image_list)
                id2tasks[rid] = tasks
    return id2tasks

def extract_boxed_text(text):
    m = re.search(r'\\boxed\{([^}]+)\}', text or "")
    return m.group(1) if m else ""

def find_node_by_id(node, target_id, level=0):
    node_id = node.get('id', None)
    if str(node_id) == str(target_id):
        return node
    for c in node.get('children', []):
        found = find_node_by_id(c, target_id, level+1)
        if found:
            return found
    return None

def check_yes_no(boxed_text):
    if boxed_text == "有":
        return True
    elif boxed_text == "无":
        return False
    else:
        return None

def find_node_by_id_for_prompt(domtree, node_id):
    if not domtree:
        return None
    if isinstance(domtree, str):
        try:
            domtree = ast.literal_eval(domtree)
        except Exception:
            print(f"[WARN] domtree字符串转dict失败，domtree:{domtree}")
            return None
    if str(domtree.get('id')) == str(node_id):
        return {
            "id": domtree.get("id"),
            "tag": domtree.get("tag"),
            "attrs": domtree.get("attrs"),
            "visible_text": domtree.get("visible_text"),
        }
    for child in domtree.get('children', []):
        res = find_node_by_id_for_prompt(child, node_id)
        if res:
            return res
    return None

def kill_process_on_port(port):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections(kind='inet'):
                if conn.laddr.port == port:
                    print(f"Killing pid={proc.pid}, name={proc.name()}, port={port}")
                    for child in proc.children(recursive=True):
                        print(f"Killing child pid={child.pid}, name={child.name()}")
                        child.kill()
                    proc.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        except Exception as e:
            print(f"Error checking proc {proc.pid}: {e}")

def clean_text(text):
    if not text:
        return ''
    return ''.join(c if c.isalnum() or c in ('-_') else '_' for c in text.strip())

def wait_server_up(url, timeout=40):
    start_time = time.time()
    while True:
        try:
            res = http_get(url, timeout=2)
            if res.status_code == 200:
                return True
        except Exception:
            pass
        if time.time() - start_time > timeout:
            raise TimeoutError("Server not up after {}s".format(timeout))

def get_current_img_b64(port):
    r = http_get(f"http://localhost:{port}/observe_sized").json()
    return r["image_b64"]

def domtree_hash(domtree):
    domtree_str = json.dumps(domtree, sort_keys=True)
    return hashlib.md5(domtree_str.encode('utf-8')).hexdigest()

def save_image(img_b64, path):
    with open(path, "wb") as f:
        f.write(base64.b64decode(img_b64))

def deep_diff(a, b):
    return json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True)

def collect_nodes_by_tag(node, tags):
    res = []
    if "tag" in node and node["tag"].lower() in tags:
        res.append({
            "id": node["id"],
            "tag": node["tag"].lower(),
            "attrs": node.get('attrs', {})
        })
    for child in node.get("children", []):
        res.extend(collect_nodes_by_tag(child, tags))
    return res

def collect_nodes_by_id(node, ids):
    res = []
    if "id" in node and node["id"] in ids:
        res.append({
            "id": node["id"],
            "attrs": node.get('attrs', {}),
            "visible_text" : node.get('visible_text', "")
        })
    for child in node.get("children", []):
        res.extend(collect_nodes_by_id(child, ids))
    return res

def collect_button_ids(node, button_text_visited=None, no_text_attrs_visited=None):
    if button_text_visited is None:
        button_text_visited = set()
    if no_text_attrs_visited is None:
        no_text_attrs_visited = set()
    res = []
    children = node.get("children", [])
    for child in children:
        if not "tag" in child:
            continue
        tag = child["tag"].lower()
        if tag == "button" or tag == "a" or (tag == "input" and child["attrs"].get("type", "").lower() in ("button", "submit")):
            txt = (child.get("attrs", {}).get("text", None) or child.get("visible_text", None) or "").strip()
            can_interact = child.get("can_interact", True)
            if can_interact:
                if txt:
                    if txt not in button_text_visited:
                        button_text_visited.add(txt)
                        res.append({
                            "id": child["id"], "text": txt, "tag": tag, "attrs": child.get("attrs", {}), "can_interact": can_interact
                        })
                else:
                    attrs_set = frozenset(child.get("attrs", {}).items())
                    if attrs_set not in no_text_attrs_visited:
                        no_text_attrs_visited.add(attrs_set)
                        res.append({
                            "id": child["id"], "tag": tag, "attrs": child.get("attrs", {}), "can_interact": can_interact
                        })
        res.extend(collect_button_ids(child, button_text_visited, no_text_attrs_visited))
    return res

def get_node_text(node):
    texts = []
    if "text" in node:
        texts.append(node["text"])
    for c in node.get("children", []):
        texts.append(get_node_text(c))
    return ''.join(texts)

def collect_select_values(node):
    results = []
    def dfs(n):
        if "tag" in n and n["tag"].lower() == "select":
            values = []
            for c in n.get("children", []):
                if c.get("tag", "").lower() == "option":
                    v = c.get("attrs", {}).get("value")
                    if v is None or v == "":
                        v = get_node_text(c)
                    values.append(v)
            results.append({"id": n["id"], "values": values})
        for c in n.get("children", []):
            dfs(c)
    dfs(node)
    return results

def set_page_state(state, port):
    http_post(f"http://localhost:{port}/reset").json()
    for step in state["actions"]:
        action = step["action"]
        dt_info = http_get(f"http://localhost:{port}/dom_tree_with_id").json()
        id2xpath = dt_info["id2xpath"]
        if action == "click":
            http_post(f"http://localhost:{port}/click",
                      json={"id": step["id"], "id2xpath": id2xpath}).json()
        elif action == "input":
            http_post(f"http://localhost:{port}/enter",
                      json={"id": step["id"], "text": step["input_val"], "id2xpath": id2xpath}).json()
        elif action == "select":
            http_post(f"http://localhost:{port}/select",
                      json={"id": step["id"], "value": step["value"], "id2xpath": id2xpath}).json()
    time.sleep(0.5)

def set_page_state_for_each_action(state, port):
    img_b64_list = []
    for step in state["actions"]:
        action = step["action"]
        dt_info = http_get(f"http://localhost:{port}/dom_tree_with_id").json()
        id2xpath = dt_info["id2xpath"]
        if action == "click":
            http_post(f"http://localhost:{port}/click",
                      json={"id": step["id"], "id2xpath": id2xpath}).json()
            img_b64 = get_current_img_b64(port)
            img_b64_list.append({"action": step, "img_b64": img_b64})
        elif action == "input":
            http_post(f"http://localhost:{port}/enter",
                      json={"id": step["id"], "text": step["input_val"], "id2xpath": id2xpath}).json()
            img_b64 = get_current_img_b64(port)
            img_b64_list.append({"action": step, "img_b64": img_b64})
        elif action == "select":
            http_post(f"http://localhost:{port}/select",
                      json={"id": step["id"], "value": step["value"], "id2xpath": id2xpath}).json()
            img_b64 = get_current_img_b64(port)
            img_b64_list.append({"action": step, "img_b64": img_b64})
    return img_b64_list

def extract_boxed_actions(text):
    return [act for act in re.findall(r'\\boxed\{([^}]*)\}', text or "") if act]

def smart_split(s, sep=','):
    res, buf, level = [], '', 0
    for c in s:
        if c == '[': level += 1
        elif c == ']': level -= 1
        if c == sep and level == 0:
            res.append(buf.strip()); buf = ''
        else:
            buf += c
    if buf: res.append(buf.strip())
    return res

def parse_action_seq(raw_action_seq):
    def strip_quotes(s):
        return re.sub(r'^[\'"“”‘’]+|[\'"“”‘’]+$', '', s)
    actions = []
    for act in smart_split(raw_action_seq):
        act = act.strip()
        if not act: continue
        m = re.match(r'click\[(.*?)\]', act, re.I)
        if m:
            id_ = strip_quotes(m.group(1).strip())
            actions.append({"action": "click", "id": id_}); continue
        m = re.match(r'enter\[(.*?)\]\[(.*?)\]', act, re.I)
        if m:
            id_ = strip_quotes(m.group(1).strip())
            val = strip_quotes(m.group(2).strip())
            actions.append({"action": "input", "id": id_, "input_val": val}); continue
        m = re.match(r'select\[(.*?)\]\[(.*?)\]', act, re.I)
        if m:
            id_ = strip_quotes(m.group(1).strip())
            val = strip_quotes(m.group(2).strip())
            actions.append({"action": "select", "id": id_, "value": val}); continue
        return None
    return actions

def node_key(node):
    if not node: return None
    return (node.get('id'), node.get('tag'), tuple(sorted(node.get('attrs', {}).items())))

def _slugify_task(task_name: str) -> str:
    if not task_name:
        return "task"
    s = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in task_name.strip())
    return s[:64] or "task"

def _extract_task_boxes_and_state(text: str):
    result = defaultdict(lambda: {"boxes": [], "state": None})
    patt = re.compile(r'\\task\{([^}]+)\}(.*?)(?=\\task\{|$)', flags=re.DOTALL)
    for task_id, segment in re.findall(patt, text or ''):
        boxes = re.findall(r'\\boxed\{([^}]*)\}', segment)
        state = None
        m = re.search(r'\\state\{([^}]*)\}', segment)
        if m:
            state = m.group(1).strip()
        result[task_id.strip()]["boxes"].extend([b.strip() for b in boxes if b is not None])
        if state:
            result[task_id.strip()]["state"] = state
    return dict(result)

def _collect_nodes_by_ids_safe(node, ids_set):
    res = []
    if not node:
        return res
    if str(node.get("id")) in ids_set:
        res.append({
            "id": node.get("id"),
            "tag": node.get("tag"),
            "attrs": node.get('attrs', {}),
            "visible_text": node.get('visible_text', "")
        })
    for c in node.get("children", []) or []:
        res.extend(_collect_nodes_by_ids_safe(c, ids_set))
    return res

def get_actions_for_tasks_by_model(img_b64, domtree, tasks, messages):
    prompt = f"""你是一位互动网页助手，我现在希望检测这个网页中的某些可互动按钮是否都可以正常工作。我会给你数个task，你需要阅读当前页面，并为每一个task选择一个动作序列，如果你认为当前页面中的内容不足以完成task，请只选择页面中有的可以完成task一部分的互动组件。
        将你的互动组件分别用latex的\\boxed{{}}包裹。动作格式：click[id]表示点击，enter[id][text]表示输入内容，select[id][text]表示选择，每个操作中间用逗号分隔。 请注意，id为此组件在domtree中的数字id
        同时在每一个boxed{{}}前面写上\\task{{本任务名称}}，并在boxed{{}}后面写上\\state{{完成}}或者 \\state{{继续}}。task列表：{str(tasks)}页面信息:\n{domtree}\n,"""
    image_content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64}}
    ]
    messages.append({"role": "user", "content": image_content})
    out = safe_llm_call(messages=messages, max_tokens=4000)
    print(out)
    messages.append({"role": "assistant", "content": out})
    mapping = _extract_task_boxes_and_state(out)

    parsed = {}
    for t in tasks:
        t_key = t
        if t_key in mapping:
            boxes = mapping[t_key]["boxes"]
            seqs = []
            for bx in boxes:
                actions = parse_action_seq(bx)
                if actions:
                    seqs.append(actions)
            parsed[t_key] = {"actions_seqs": seqs, "raw_output": out, "state": mapping[t_key]["state"]}
        else:
            parsed[t_key] = {"actions_seqs": [], "raw_output": out, "state": None}
    return parsed

def get_next_actions_for_single_task(img_b64, domtree, task_text, messages):
    prompt = f"""
        你是一位互动网页助手，我现在希望你完成一个task。你目前正处于检测过程中，请阅读历史页面上已经点击了哪些按钮。请只选择页面上可以完成点击的按钮。你只专注于完成一个task，你需要阅读当前页面，并为本进行中task选择一个进行中动作序列，如果你认为当前页面中的内容不足以完成task，请只选择页面中有的可以完成task一部分的互动组件。
        将你的互动组件用一个latex的\\boxed{{}}包裹。动作格式：click[id]表示点击，enter[id][text]表示输入内容，select[id][text]表示选择，每个操作中间用逗号分隔。请注意，id为此组件在domtree中的数字id
        同时在每一个boxed{{}}前面写上\\task{{本任务名称}}，并在boxed{{}}后面写上\\state{{完成}}或者 \\state{{继续}}。task内容：{task_text}页面信息:\n{domtree}\n,"""
    image_content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64}}
    ]
    messages.append({"role": "user", "content": image_content})
    out = safe_llm_call(messages=messages, max_tokens=3000)
    messages.append({"role": "assistant", "content": out})

    boxes = re.findall(r'\\boxed\{([^}]*)\}', out or "")
    actions = None
    for bx in boxes:
        cand = parse_action_seq(bx)
        if cand is not None:
            actions = cand
            break
    st = None
    m = re.search(r'\\state\{([^}]*)\}', out or "")
    if m:
        st = m.group(1).strip()
    return actions, out, st

def verify_single_task_with_history(img_b64_list_ordered, task_text, messages_chain):
    prompt = (
        "给出这一路径的所有截图（按时间正序）。任务如下：\n"
        f"【任务】{task_text}\n\n"
        "请判断是否已经完成了该任务应有的网页变化。"
        "请注意！每个task名称仅代表一个互动按钮，比如“新建”仅代表打开新建页面，不代表保存。所以只需要检测新建页面是否能打开。有且仅有在task中明说“新建-输入。。。-保存”才需要验证保存。同理，“删除”也仅代表打开删除框，“删除-确认删除”才代表完成了两次操作，才需要检测是否真的完成了删除。"
        "若完成，请在 \\boxed{有}；若未完成，\\boxed{无}。"
        "随后简要说明理由。"
    )
    content = [{"type": "text", "text": prompt}]
    for img in img_b64_list_ordered:
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}})
    messages_chain.append({"role": "user", "content": content})
    out = safe_llm_call(messages=messages_chain, max_tokens=4000)
    messages_chain.append({"role": "assistant", "content": out})
    bx = extract_boxed_text(out)
    flag = check_yes_no(bx)
    return (flag is True), out

def _exec_actions_and_collect_images(port, actions):
    out = []
    resplist = []
    if not actions:
        return out, resplist
    for a in actions:
        dt_info = http_get(f"http://localhost:{port}/dom_tree_with_id").json()
        id2xpath = dt_info["id2xpath"]
        try:
            if a["action"] == "click":
                resp = http_post(f"http://localhost:{port}/click",
                                 json={"id": a["id"], "id2xpath": id2xpath})
            elif a["action"] == "input":
                resp = http_post(f"http://localhost:{port}/enter",
                                 json={"id": a["id"], "text": a.get("input_val",""), "id2xpath": id2xpath})
            elif a["action"] == "select":
                resp = http_post(f"http://localhost:{port}/select",
                                 json={"id": a["id"], "value": a.get("value",""), "id2xpath": id2xpath})
            else:
                continue
            try:
                data = resp.json()
                resplist.append(data.get("result"))
            except ValueError:
                resplist.append(None)
                print("POST resp not JSON:", resp.status_code, resp.text[:200])
        except requests.RequestException as e:
            print(f"[ERROR] action HTTP error: {e}")
            resplist.append(False)

        try:
            img_b64 = get_current_img_b64(port)
            out.append({"action": a, "img_b64": img_b64})
        except Exception as e:
            print(f"[ERROR] get image after action failed: {e}")
            resplist.append(False)
    return out, resplist

def multi_task_explore_html(html_file, port, tasks, max_rounds=6, per_branch_hard_timeout=1000):
    step_records = []
    result_per_task = {}

    html_name = os.path.splitext(os.path.basename(html_file))[0]
    base_dir = f"./{IMAGE_DIR}/{html_name}"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(BUG_DIR, exist_ok=True)

    # 启动 webenv
    kill_process_on_port(port)
    p = subprocess.Popen([sys.executable, WEBENV, html_file, '--port', str(port)])
    try:
        wait_server_up(f"http://localhost:{port}/dom_tree_with_id")
        dt0 = http_get(f"http://localhost:{port}/dom_tree_with_id").json()
        dom0 = dt0["domtree"]
        start_b64 = get_current_img_b64(port)

        start_path = os.path.join(base_dir, "start.png")
        save_image(start_b64, start_path)

        # 首轮：一次请求拿所有 task 的起步动作
        multi_msgs = []
        print("正在获取 first")
        tmap = get_actions_for_tasks_by_model(start_b64, dom0, tasks, multi_msgs)

        branches = []
        for t in tasks:
            slug = _slugify_task(t)
            t_dir = os.path.join(base_dir, slug)
            os.makedirs(t_dir, exist_ok=True)

            branch = {
                "task": t,
                "slug": slug,
                "dir": t_dir,
                "messages": [],
                "images": [start_b64],
                "image_files": [start_path],
                "actions": [],
                "raw_outputs": [],
                "done": False,
                "start_time": time.time()
            }
            seqs = tmap.get(t, {}).get("actions_seqs", [])
            if not seqs:
                cp_bad_html(
                    html_file,
                    reason="no_first_seq_from_model",
                    task=t,
                    step={"phase": "first_round_get_actions"},
                    domtree=dom0,
                    img_b64=start_b64,
                    model_output=tmap.get(t, {}).get("raw_output", ""),
                    inputs={"tasks_all": tasks}
                )

            if seqs:
                first_seq = seqs[0]
                http_post(f"http://localhost:{port}/reset").json()
                step_imgs, resplist = _exec_actions_and_collect_images(port, first_seq)
                for resp in resplist:
                    if resp is False:
                        print(f"ERROR: error performing action: {first_seq}")
                        cp_bad_html(
                            html_file,
                            reason="first_seq_action_failed",
                            task=t,
                            step={"phase": "first_round_exec", "after_index": len(step_imgs)},
                            actions=first_seq,
                            responses=resplist,
                            img_b64=step_imgs[-1]["img_b64"] if step_imgs else start_b64,
                            model_output=tmap.get(t, {}).get("raw_output", "")
                        )
                texts = ["start"]
                for idx, s in enumerate(step_imgs, 1):
                    a = s["action"]
                    tag = f"{a['action']}_{a['id']}"
                    if a["action"] == "input": tag += f"_{a.get('input_val','')}"
                    if a["action"] == "select": tag += f"_{a.get('value','')}"
                    texts.append(clean_text(tag))
                    seq_name = "_".join(texts)
                    fp = os.path.join(t_dir, f"{seq_name}.png")
                    save_image(s["img_b64"], fp)

                    branch["images"].append(s["img_b64"])
                    branch["image_files"].append(fp)
                    branch["actions"].append(a)

                step_records.append({
                    "step_type": "list",
                    "img_name": "start",
                    "task": t,
                    "domtree": str(dom0),
                    "actions_seqs": [dict(a) for a in first_seq],
                    "model_output": tmap.get(t, {}).get("raw_output", ""),
                    "step_idx": len(step_records),
                })
                branch["raw_outputs"].append(tmap.get(t, {}).get("raw_output", ""))

            st = tmap.get(t, {}).get("state")
            print(st)
            if st == "完成":
                branch["done"] = True

            branches.append(branch)

        # 中间轮推进
        for rnd in range(1, max_rounds):
            if all(b["done"] for b in branches):
                break
            for br in branches:
                if br["done"]:
                    continue
                if time.time() - br["start_time"] > per_branch_hard_timeout:
                    print(f"[WARN] branch {br['task']} exceeded {per_branch_hard_timeout}s, mark done.")
                    br["done"] = True
                    continue

                print("正在开启第二轮")
                http_post(f"http://localhost:{port}/reset").json()
                if br["actions"]:
                    dt_info = http_get(f"http://localhost:{port}/dom_tree_with_id").json()
                    id2xpath = dt_info["id2xpath"]
                    for a in br["actions"]:
                        try:
                            if a["action"] == "click":
                                resp = http_post(f"http://localhost:{port}/click", json={"id": a["id"], "id2xpath": id2xpath}).json()
                            elif a["action"] == "input":
                                resp = http_post(f"http://localhost:{port}/enter", json={"id": a["id"], "text": a.get("input_val",""), "id2xpath": id2xpath}).json()
                            elif a["action"] == "select":
                                resp = http_post(f"http://localhost:{port}/select", json={"id": a["id"], "value": a.get("value",""), "id2xpath": id2xpath}).json()
                            else:
                                resp = {"result": True}
                            if resp.get("result") is False:
                                print(f"ERROR: Failed performing action: {a}")
                                cp_bad_html(
                                    html_file,
                                    reason="replay_action_result_false",
                                    task=br["task"],
                                    step={"phase": "replay_history_actions"},
                                    actions=[a],
                                    responses=[resp],
                                    domtree=http_get(f"http://localhost:{port}/dom_tree_with_id").json().get("domtree"),
                                    img_b64=get_current_img_b64(port)
                                )
                        except Exception as e:
                            print(f"[ERROR] replay action failed: {e}")
                            cp_bad_html(
                                html_file,
                                reason="replay_action_exception",
                                task=br["task"],
                                step={"phase": "replay_history_actions"},
                                actions=[a],
                                exception=e,
                                domtree=http_get(f"http://localhost:{port}/dom_tree_with_id").json().get("domtree"),
                                img_b64=get_current_img_b64(port)
                            )

                dt_now = http_get(f"http://localhost:{port}/dom_tree_with_id").json()
                dom_now = dt_now["domtree"]
                img_now = get_current_img_b64(port)

                next_actions, raw_out, st = get_next_actions_for_single_task(img_now, dom_now, br["task"], br["messages"])
                print(raw_out)
                br["raw_outputs"].append(raw_out)

                if not next_actions:
                    br["done"] = True
                    continue

                step_imgs, resplist = _exec_actions_and_collect_images(port, next_actions)
                for resp in resplist:
                    if resp is False:
                        cp_bad_html(
                            html_file,
                            reason="no_next_actions_from_model",
                            task=br["task"],
                            step={"phase": "next_round_plan"},
                            domtree=dom_now,
                            img_b64=img_now,
                            model_output=raw_out
                        )
                texts = [os.path.splitext(os.path.basename(br["image_files"][0]))[0]]
                for idx, s in enumerate(step_imgs, 1):
                    a = s["action"]
                    tag = f"{a['action']}_{a['id']}"
                    if a["action"] == "input": tag += f"_{a.get('input_val','')}"
                    if a["action"] == "select": tag += f"_{a.get('value','')}"
                    texts.append(clean_text(tag))
                    seq_name = "_".join(texts)
                    fp = os.path.join(br["dir"], f"{seq_name}.png")
                    save_image(s["img_b64"], fp)

                    br["images"].append(s["img_b64"])
                    br["image_files"].append(fp)
                    br["actions"].append(a)

                if st == "完成":
                    br["done"] = True

        # 末尾统一 verify
        for br in branches:
            verify_msgs = []
            try:
                ok, verify_out = verify_single_task_with_history(br["images"], br["task"], verify_msgs)
            except Exception as e:
                print(f"[ERROR] verify failed: {e}")
                cp_bad_html(
                    html_file,
                    reason="verify_exception",
                    task=br["task"],
                    step={"phase": "verify"},
                    exception=e,
                    inputs={"images": br["image_files"]},
                    extra={"verify_msgs_len": len(verify_msgs)}
                )
                ok, verify_out = False, f"verify_exception: {e}"
            result_per_task[br["task"]] = {
                "pass": bool(ok),
                "task": br["task"],
                "images": br["image_files"],
                "actions": br["actions"],
                "model_raw_outputs": br["raw_outputs"],
                "verify_output": verify_out
            }

        out_json = {
            "html_file": html_file,
            "mode": "multi_task_branch_then_verify",
            "tasks": [b["task"] for b in branches],
            "results_per_task": result_per_task
        }
        out_path = os.path.join(f"./{IMAGE_DIR}/", f"{html_name}__tasks.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)
        print(f"[OK] 多 task 结果已保存：{out_path}")

    finally:
        try:
            p.terminate()
        except Exception:
            pass
        kill_process_on_port(port)

def chunkify(lst, n):
    if n <= 0: return [lst]
    return [lst[i::n] for i in range(n)]

def worker(html_list, port, progress_q, id2tasks, config: RunnerConfig):
    # --- 子进程内重建 requests 会话（spawn/fork 安全） ---
    global SESSION
    SESSION = requests.Session()
    SESSION.trust_env = False

    # --- 子进程内也要绑定 OpenAI 客户端与全局别名 ---
    try:
        local_client = OpenAI(api_key=config.api_key, base_url=config.api_base)
        _bind_config_aliases(config, local_client)
    except Exception as e:
        print(f"[ERROR] worker init _bind_config_aliases failed on port {port}: {e}")

    for html in html_list:
        try:
            html_name = os.path.basename(html)
            html_id = os.path.splitext(html_name)[0]
            json_log_path = os.path.join(f"./{IMAGE_DIR}/", f"{html_id}__tasks.json")
            if os.path.exists(json_log_path):
                print(f"[SKIP] 已存在日志: {json_log_path}，跳过 {html}")
                continue
            tasks = id2tasks.get(html_id, [])
            if not tasks:
                print(f"[WARN] 未找到 id={html_id} 的 tasks，跳过 {html}")
                continue
            print(f"[RUN] {html_name} → tasks={tasks}")
            # 这里依旧使用全局别名（已在本子进程重新注入）
            multi_task_explore_html(html, port, tasks)
        except Exception as e:
            print(f"[ERROR] worker on {html} failed: {e}")
        finally:
            try:
                progress_q.put(1, block=False)
            except Exception:
                pass


# =========================================================
# =========================== main ========================
# =========================================================
if __name__ == '__main__':
    # 0) 加载配置（dataclass）
    CONFIG, _args = load_config_or_exit()

    # 1) 初始化 OpenAI 客户端 + 绑定兼容别名
    CLIENT = OpenAI(api_key=CONFIG.api_key, base_url=CONFIG.api_base)
    _bind_config_aliases(CONFIG, CLIENT)

    # 2) 校验/加载 tasks
    if not os.path.exists(TASKS_SOURCE_JSONL):
        print(f("[ERROR] TASKS_SOURCE_JSONL 不存在：{TASKS_SOURCE_JSONL}"))
        sys.exit(1)
    id2tasks = load_id2tasks_from_jsonl(TASKS_SOURCE_JSONL, INPUT_TYPE)
    print(f"[OK] 已加载 tasks 条目：{len(id2tasks)}（模式={INPUT_TYPE}）")

    # 3) 列出 HTML
    if not os.path.exists(input_dir):
        print(f"[ERROR] HTML 目录不存在：{input_dir}")
        sys.exit(1)
    html_files = [
        os.path.join(input_dir, fn)
        for fn in os.listdir(input_dir)
        if fn.endswith('.html')
    ]
    if not html_files:
        print(f"[WARN] 目录无 HTML：{input_dir}")
        sys.exit(0)

    # 4) 并发调度
    groups = chunkify(html_files, NUM_PORT)
    ports = [PORT_BASE + i for i in range(NUM_PORT)]
    progress_q = Queue()
    total = len(html_files)
    pbar = tqdm(total=total, desc='全部HTML', ncols=80)
    procs = []
    for group, port in zip(groups, ports):
        p = Process(target=worker, args=(group, port, progress_q, id2tasks, CONFIG))
        p.daemon = True
        p.start()
        procs.append(p)
        time.sleep(1.0)

    finished = 0
    while finished < total:
        try:
            n = progress_q.get(timeout=40)
            finished += n
            pbar.update(n)
        except QueueEmpty:
            alive = any(p.is_alive() for p in procs)
            if not alive:
                print("\n[WARN] 子进程已全部退出，但进度不足 total，视为部分任务失败。")
                break

    for p in procs:
        p.join(timeout=5)
    pbar.close()
    print(f"[DONE] 期望 {total} 个 HTML，实际完成/记账 {finished} 个。")
