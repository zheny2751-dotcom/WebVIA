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
from multiprocessing import Process, Queue
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

# ===================== 手动配置（全局默认，不再直接用于输出目录） =====================
IMAGE_DIR = ""   # 仅作为默认值；实际每个输入目录会派生自己的
input_dir = ''
OUTPUT_ROOT = "./verify_results"
TASKS_SOURCE_JSONL = 'input_data/data.jsonl'
MODEL_NAME = "gpt-5-2025-08-07"
BUG_DIR = ""


INPUT_DIRS = [
    "outputs/o4-mini-2025-04-16_html",
    "outputs/claude-sonnet-4-20250514-thinking_html",
    "outputs/gpt-5-2025-08-07_html",
    "outputs/gemini-2.5-pro_html",
    "outputs/claude-3-7-sonnet-20250219-thinking_html",
    "outputs/gpt-4o-2024-11-20_html"
]

# 并发端口设置（更稳）：最多 8 或 CPU 核心数
NUM_PORT = 40
PORT_BASE = 8000

# ===== 从配置文件加载 OpenAI 参数 =====
def load_api_config(config_path="api_config.json"):
    """从 JSON 文件加载 api_key 和 api_base"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "api_key" not in cfg or "api_base" not in cfg:
        raise ValueError("配置文件必须包含 'api_key' 和 'api_base' 字段")
    return cfg["api_base"], cfg["api_key"]

# ===== 加载配置 =====
api_base, api_key = load_api_config("api_config.json")

openai.api_base = api_base
openai.api_key = api_key


client = OpenAI(api_key=openai.api_key, base_url=openai.api_base)


# ================= 网络/LLM 超时与会话封装 =================
DEFAULT_TIMEOUT = 60            # 每次本地 HTTP 超时（秒）
MODEL_TIMEOUT = 1000            # 单次 LLM 推理超时（秒）
MODEL_MAX_TOKENS = 10000

# 全局 requests.Session，避免系统代理干扰
SESSION = requests.Session()
SESSION.trust_env = False

def http_get(url, timeout=DEFAULT_TIMEOUT, **kw):
    r = SESSION.get(url, timeout=timeout, **kw)
    r.raise_for_status()
    return r

def http_post(url, timeout=DEFAULT_TIMEOUT, **kw):
    r = SESSION.post(url, timeout=timeout, **kw)
    r.raise_for_status()
    return r

def _call_llm(messages, max_tokens):
    return client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=max_tokens,
    )

def safe_llm_call(messages, max_tokens=MODEL_MAX_TOKENS, timeout=MODEL_TIMEOUT):
    """线程池套一层超时，防止底层 SDK 卡住。"""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_call_llm, messages, max_tokens)
        try:
            resp = fut.result(timeout=timeout)
            return (resp.choices[0].message.content or "").strip()
        except FuturesTimeout:
            raise TimeoutError(f"LLM call exceeded {timeout}s")

# -------------------- 工具函数：tasks 生成 --------------------

ACTION_MARKERS = {"input", "click", "select"}
BUTTON_WORDS = {
    "添加成就", "删除", "编辑", "保存成就", "保存", "确认", "提交", "关闭", "取消"
}

import traceback
from pathlib import Path

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
    """确保 original.html 只复制一次。"""
    try:
        dst = html_dir / f"original{ext}"
        if not dst.exists():
            shutil.copy(html_src_path, dst)
    except Exception as e:
        print(f"[BUG-LOG] copy html failed: {e}")

_idx_pat = re.compile(r"_(\d{3})\.")

def _next_issue_index(html_dir: Path) -> int:
    """
    在 html_dir 中扫描现有编号，返回下一个编号。
    会查看 bug_XXX.json、domtree_XXX.json/txt、input_XXX.png 等文件。
    """
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
    step: dict | None = None,          # 例如 {"phase": "...", "step_idx": n}
    actions: list | None = None,       # list[dict] 动作序列
    responses: list | None = None,     # 与动作对齐的返回
    domtree: dict | str | None = None, # DOM 树（dict 或 str）
    img_b64: str | None = None,        # 当时截图
    model_output: str | None = None,   # 模型原文输出
    inputs: dict | None = None,        # 额外输入上下文（tasks、参数等）
    exception: Exception | str | None = None,  # 异常对象或文本
    extra: dict | None = None,         # 其他自定义字段
    bug_dir: str | None = None,        # ★ 新增：指定本目录的 BUG_DIR
):
    """
    将同一个 HTML 的所有错误放入同一目录：
      {bug_dir or BUG_DIR}/{html_stem}/
        ├─ original.html
        ├─ bug_log.jsonl
        ├─ bug_000.json
        ├─ domtree_000.json/txt
        ├─ input_000.png
        ├─ model_output_000.txt
        └─ exception_000.txt
    """
    basename = os.path.basename(src_file)
    stem, ext = os.path.splitext(basename)
    base_bug_dir = bug_dir or BUG_DIR
    html_dir = Path(base_bug_dir) / stem
    html_dir.mkdir(parents=True, exist_ok=True)

    # original.html 只保存一次
    _ensure_original_html(src_file, html_dir, ext)

    # 分配本次问题的自增编号
    idx = _next_issue_index(html_dir)
    idx3 = f"{idx:03d}"

    print(f"正在记录 BUG：{src_file} | 原因={reason or 'unspecified'} | 索引={idx3}")

    # —— 写入结构化记录（单条）
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

    # —— 追记总日志（jsonl）
    try:
        with open(html_dir / "bug_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[BUG-LOG] append bug_log.jsonl failed: {e}")

    # —— DOM
    if domtree is not None:
        if isinstance(domtree, (dict, list)):
            _safe_write_json(html_dir / f"domtree_{idx3}.json", domtree)
        else:
            _safe_write_text(html_dir / f"domtree_{idx3}.txt", domtree)

    # —— 截图
    if img_b64:
        _safe_save_b64_png(img_b64, html_dir / f"input_{idx3}.png")

    # —— 模型输出
    if model_output:
        _safe_write_text(html_dir / f"model_output_{idx3}.txt", model_output)

    # —— 异常
    if exception:
        if isinstance(exception, Exception):
            tb = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
            _safe_write_text(html_dir / f"exception_{idx3}.txt", tb)
        else:
            _safe_write_text(html_dir / f"exception_{idx3}.txt", str(exception))

def _basename_stem(p: str) -> str:
    """从路径取出无扩展名的文件基名。"""
    name = os.path.basename(p)
    if name.endswith(".png"):
        name = name[:-4]
    return name

def _collapse_underscores(s: str) -> str:
    """把多个下划线折叠为一个（https___example → https_example）。"""
    return re.sub(r"_+", "_", s)

def _parse_stem_to_task(stem: str) -> str:
    """
    把文件名stem解析成 task 文案：
      - 去掉前导 '_' / 'start'
      - 识别 input_<id>_<value...> → 'input<value>'
      - 中文按钮词直接输出；'保存成就' → '—保存成功'
      - 其它 token 用 '-' 连接
    """
    if stem == "start":
        return ""
    s = stem.lstrip("_")
    s = _collapse_underscores(s)
    tokens = [t for t in s.split("_") if t != ""]
    parts = []
    i = 0
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
    """优先依据 operation_info 生成 tasks（最可信）。"""
    tasks = []
    for op in op_info:
        end_path = op.get("end_image_path") or ""
        stem = _basename_stem(end_path)
        task = _parse_stem_to_task(stem)
        if task:
            tasks.append(task)
    # 去重保序
    seen = set(); uniq = []
    for t in tasks:
        if t not in seen:
            uniq.append(t); seen.add(t)
    return uniq

def _tasks_from_image_list(image_list):
    """
    回退方案：仅依据 image_list（常见为 6 张）。
    规则：选“不是其它图片前缀”的最后图片作为一个 task；
    同时强制保留含“_保存成就”的节点。
    """
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

def load_id2tasks_from_jsonl(jsonl_path: str):
    """
    读取 JSONL，每行构造 {id: [tasks...]}。
    优先用 operation_info；否则从 image_list 智能切割。
    """
    id2tasks = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = str(rec.get("id") or rec.get("html_id") or rec.get("name") or "").strip()
            if not rid:
                continue
            op_info = rec.get("operation_info") or []
            image_list = rec.get("image_list") or []
            if op_info:
                tasks = _tasks_from_operation_info(op_info)
            else:
                tasks = _tasks_from_image_list(image_list)
            id2tasks[rid] = tasks
    return id2tasks

# -------------------- 你的原始/改进流程 --------------------

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
    # 杀死所有监听这个端口的进程（包括子进程）
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
        res.extend(collect_nodes_by_id(child, ids))  # ★ 修正递归函数
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
        将你的互动组件分别用latex的\\boxed{{}}包裹。动作格式：click[id]表示点击，enter[id][text]表示输入内容，select[id][text]表示选择，每个操作中间用逗号分隔。 请注意，id为此组件在domtree中的id
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
        将你的互动组件用一个latex的\\boxed{{}}包裹。动作格式：click[id]表示点击，enter[id][text]表示输入内容，select[id][text]表示选择，每个操作中间用逗号分隔。
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
    """
    执行动作序列，并返回每步后的截图
    """
    out = []
    resplist = []
    if not actions:
        return out, resplist
    for a in actions:
        # ★ 每步前刷新一次 id2xpath，避免 DOM 变化导致失配
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

def multi_task_explore_html(html_file, port, tasks, image_dir, bug_dir, max_rounds=6, per_branch_hard_timeout=1000):
    """
    多 task 分叉 → 各自推进 → 末尾统一 verify。
    结果与截图写入 ./{image_dir}/{html_name}/{task_slug}/...
    汇总 JSON: ./{image_dir}/{html_name}__tasks.json
    """
    step_records = []
    result_per_task = {}

    html_name = os.path.splitext(os.path.basename(html_file))[0]
    base_dir = f"./{image_dir}/{html_name}"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(bug_dir, exist_ok=True)

    # 启动 webenv
    kill_process_on_port(port)
    p = subprocess.Popen([sys.executable, "./webenv-init/webenv.py", html_file, '--port', str(port)])
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
                    inputs={"tasks_all": tasks},
                    bug_dir=bug_dir
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
                            model_output=tmap.get(t, {}).get("raw_output", ""),
                            bug_dir=bug_dir
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
                # 分支硬超时：防止单 task 无限拖
                if time.time() - br["start_time"] > per_branch_hard_timeout:
                    print(f"[WARN] branch {br['task']} exceeded {per_branch_hard_timeout}s, mark done.")
                    br["done"] = True
                    continue

                print("正在开启第二轮")
                http_post(f"http://localhost:{port}/reset").json()
                if br["actions"]:
                    for a in br["actions"]:
                        dt_info = http_get(f"http://localhost:{port}/dom_tree_with_id").json()
                        id2xpath = dt_info["id2xpath"]
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
                                    # 当前页截图/DOM
                                    domtree=http_get(f"http://localhost:{port}/dom_tree_with_id").json().get("domtree"),
                                    img_b64=get_current_img_b64(port),
                                    bug_dir=bug_dir
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
                                img_b64=get_current_img_b64(port),
                                bug_dir=bug_dir
                            )

                dt_now = http_get(f"http://localhost:{port}/dom_tree_with_id").json()
                dom_now = dt_now["domtree"]
                img_now = get_current_img_b64(port)

                next_actions, raw_out, st = get_next_actions_for_single_task(img_now, dom_now, br["task"], br["messages"])
                print(raw_out)
                br["raw_outputs"].append(raw_out)

                if not next_actions:
                    br["done"] = True  # ★ 无下一步动作，结束该分支（防止死循环）
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
                            model_output=raw_out,
                            bug_dir=bug_dir
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
                    extra={"verify_msgs_len": len(verify_msgs)},
                    bug_dir=bug_dir
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
            step_records.append({
                "step_type": "compare",
                "task": br["task"],
                "img_name_before": "start",
                "img_name_list_after": br["image_files"][1:],
                "domtree": "N/A in final compare (use per-step if needed)",
                "actions_seq": [dict(a) for a in br["actions"]],
                "component_info": [],
                "model_output": "\n".join(br["raw_outputs"]),
                "detect_output": verify_out,
                "pass": bool(ok),
                "step_idx": len(step_records),
            })

        out_json = {
            "html_file": html_file,
            "mode": "multi_task_branch_then_verify",
            "tasks": tasks,
            "steps": step_records,
            "results_per_task": result_per_task
        }
        out_path = os.path.join(f"./{image_dir}/", f"{html_name}__tasks.json")
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

# ===================== 并发入口（整合 id→tasks） =====================

def chunkify(lst, n):
    """均分列表lst为n组，返回二维列表"""
    if n <= 0: return [lst]
    return [lst[i::n] for i in range(n)]

def worker(html_list, port, progress_q, id2tasks, image_dir, bug_dir):
    for html in html_list:
        try:
            html_name = os.path.basename(html)
            html_id = os.path.splitext(html_name)[0]  # 如 1009
            json_log_path = os.path.join(f"./{image_dir}/", f"{html_id}__tasks.json")  # ★ 与输出统一

            if os.path.exists(json_log_path):
                print(f"[SKIP] 已存在日志: {json_log_path}，跳过 {html}")
                continue

            tasks = id2tasks.get(html_id, [])
            if not tasks:
                print(f"[WARN] 未找到 id={html_id} 的 tasks，跳过 {html}")
                continue

            print(f"[RUN] {html_name} → tasks={tasks}")
            multi_task_explore_html(html, port, tasks, image_dir=image_dir, bug_dir=bug_dir)

        except Exception as e:
            print(f"[ERROR] worker on {html} 失败: {e}")

        finally:
            # **无论成功/失败都回填一次**，确保主进度不会因为异常而卡住
            try:
                progress_q.put(1, block=False)
            except Exception:
                pass

# ===================== 多目录顺序执行的封装 =====================

def _derive_dirs_from_input_dir(dir_path: str) -> tuple[str, str]:
    """根据输入目录名自动生成 IMAGE_DIR 与 BUG_DIR。"""
    base = os.path.basename(os.path.normpath(dir_path))
    image_dir = f"{OUTPUT_ROOT}/Verify_{base}"
    bug_dir   = f"{OUTPUT_ROOT}/BUG_verify_{base}"
    return image_dir, bug_dir

def process_one_dir(dir_path: str, id2tasks: dict):
    """顺序处理一个 HTML 目录；本函数会阻塞到该目录所有 HTML 跑完为止。"""
    if not os.path.exists(dir_path):
        print(f"[ERROR] HTML 目录不存在：{dir_path}")
        return

    html_files = [
        os.path.join(dir_path, fn)
        for fn in os.listdir(dir_path)
        if fn.endswith('.html')
    ]
    if not html_files:
        print(f"[WARN] 目录无 HTML：{dir_path}")
        return

    # ★ 目录专属输出路径
    image_dir, bug_dir = _derive_dirs_from_input_dir(dir_path)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(bug_dir, exist_ok=True)

    # 并发调度（与原逻辑相同）
    groups = chunkify(html_files, NUM_PORT)
    ports = [PORT_BASE + i for i in range(NUM_PORT)]
    progress_q = Queue()
    total = len(html_files)
    pbar = tqdm(total=total, desc=f'目录 {os.path.basename(dir_path)}', ncols=80)

    procs = []
    for group, port in zip(groups, ports):
        p = Process(target=worker, args=(group, port, progress_q, id2tasks, image_dir, bug_dir))
        p.daemon = True
        p.start()
        procs.append(p)
        time.sleep(1.0)  # 启动间隔，降低竞争

    finished = 0
    # 主循环：带超时与健康检查，防止“永远等不到”
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

    # 清理
    for p in procs:
        try:
            p.join(timeout=5)
        except Exception:
            pass
    pbar.close()
    print(f"[DONE] 目录 {dir_path} 期望 {total} 个 HTML，实际完成/记账 {finished} 个。")

    # 保险：目录跑完后，释放所有可能的端口占用
    for port in ports:
        kill_process_on_port(port)

# ===================== 主入口 =====================

if __name__ == '__main__':
    # 想顺序执行的多个输入目录（按需填写/增删；将逐个执行）
    # 1) 加载 id→tasks 映射（一次）
    if not os.path.exists(TASKS_SOURCE_JSONL):
        print(f"[ERROR] TASKS_SOURCE_JSONL 不存在：{TASKS_SOURCE_JSONL}")
        sys.exit(1)
    id2tasks = load_id2tasks_from_jsonl(TASKS_SOURCE_JSONL)
    print(f"[OK] 已加载 tasks 条目：{len(id2tasks)}")

    # 2) 顺序处理每个目录（每个目录自动生成自己的 IMAGE_DIR/BUG_DIR）
    for idx, dir_path in enumerate(INPUT_DIRS, 1):
        print("\n" + "=" * 80)
        print(f"[RUN] 开始处理目录 {idx}/{len(INPUT_DIRS)} ：{dir_path}")
        print("=" * 80)
        try:
            process_one_dir(dir_path, id2tasks)
        except Exception as e:
            print(f"[ERROR] 处理目录失败：{dir_path} | {e}")
        finally:
            # 收尾兜底：再清理一次端口
            for port in [PORT_BASE + i for i in range(NUM_PORT)]:
                kill_process_on_port(port)

    print("\n[ALL DONE] 所有目录已按顺序处理完成。")
