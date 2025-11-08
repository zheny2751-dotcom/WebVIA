# -*- coding: utf-8 -*-
import os
import base64
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from multiprocessing import Process, Queue, cpu_count
import argparse
import re
from openai import OpenAI

# ===================== 配置对象 =====================
@dataclass(frozen=True)
class RunnerConfig:
    input_jsonl: str
    output_prefix: str
    models: List[str]
    num_workers: int
    images_root: str
    output_html_prefix: str
    api_base: str
    api_key: str
    max_completion_tokens: int = 32000
    request_timeout: int = 600  # seconds

# ===================== 配置加载 =====================
def load_config_or_exit() -> Tuple[RunnerConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="UI2Code multi-model runner (with HTML extraction)")
    parser.add_argument("--config", required=True, help="path to config.json")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] config file not found: {args.config}")
        raise SystemExit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    required = ["input_jsonl", "output_prefix", "models"]
    missing = [k for k in required if k not in cfg or cfg[k] in (None, "")]
    if missing:
        print(f"[ERROR] missing required fields in config: {missing}")
        raise SystemExit(1)

    # 组装显式配置
    input_jsonl = cfg["input_jsonl"]
    output_prefix = cfg["output_prefix"]
    models = list(cfg["models"]) if isinstance(cfg["models"], list) else [cfg["models"]]
    num_workers = int(cfg.get("num_workers", max(1, cpu_count())))
    if num_workers <= 0:
        num_workers = max(1, cpu_count())

    # images_root 未指定时，默认使用 input_jsonl 所在目录
    images_root = cfg.get("images_root") or os.path.dirname(os.path.abspath(input_jsonl))
    output_html_prefix = cfg.get("output_html_prefix", "")
    api_base = cfg.get("api_base", "") or ""
    api_key = cfg.get("api_key", "") or ""
    max_tokens = int(cfg.get("max_completion_tokens", 10000))
    timeout = int(cfg.get("timeout", 600))

    rc = RunnerConfig(
        input_jsonl=input_jsonl,
        output_prefix=output_prefix,
        models=models,
        num_workers=num_workers,
        images_root=images_root,
        output_html_prefix=output_html_prefix,
        api_base=api_base,
        api_key=api_key,
        max_completion_tokens=max_tokens,
        request_timeout=timeout,
    )
    return rc, args

# ===================== 正则与提取工具 =====================
_CONTENT_PAT = re.compile(r"content='(.*?)',\s*refusal=None", re.DOTALL)
_CONTENT_PAT_DBL = re.compile(r'content="(.*?)",\s*refusal=None', re.DOTALL)
_HTML_BLOCK_PAT = re.compile(r'<html.*?>.*?</html>', re.DOTALL | re.IGNORECASE)

def _strip_code_fences(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    while lines and lines[0].strip().lower() in ("```html", "```"):
        lines = lines[1:]
    while lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _extract_inside_html_tag(text: str) -> str:
    if not text:
        return ""
    m = _HTML_BLOCK_PAT.search(text)
    return (m.group(0) if m else text).strip()

def _unescape_common_backslashes(s: str) -> str:
    if not s:
        return ""
    s = s.replace('\\\\', '\\')
    s = s.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
    s = s.replace('\\"', '"').replace("\\'", "'")
    return s

def _extract_content_from_mllm_generate(raw: str) -> str:
    if not raw:
        return ""
    m = _CONTENT_PAT.search(raw) or _CONTENT_PAT_DBL.search(raw)
    return m.group(1) if m else ""

def extract_html_from_any(record: Dict[str, Any]) -> str:
    txt = (record.get("output") or "").strip()
    if not txt:
        mg = record.get("mllm_generate", "")
        txt = _extract_content_from_mllm_generate(mg)
    if not txt:
        return ""
    txt = _strip_code_fences(txt)
    txt = _extract_inside_html_tag(txt)
    txt = _unescape_common_backslashes(txt)
    return txt.strip()

# ===================== 路径辅助（显式传参） =====================
def resolve_html_dir_for_model(cfg: RunnerConfig, model_name: str) -> str:
    if cfg.output_html_prefix:
        html_root = os.path.abspath(cfg.output_html_prefix)
    else:
        html_root = os.path.abspath(cfg.output_prefix)
    out_dir = os.path.join(html_root, f"{model_name}_html")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

# ===================== LLM 调用（显式传参） =====================
def make_client(cfg: RunnerConfig) -> OpenAI:
    # 使用 base_url + api_key 显式创建 client
    return OpenAI(api_key=cfg.api_key, base_url=cfg.api_base) if (cfg.api_key or cfg.api_base) else OpenAI()

def get_response(client: OpenAI, messages: List[Dict[str, Any]], model_name: str, cfg: RunnerConfig) -> str:
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=cfg.max_completion_tokens,
        timeout=cfg.request_timeout
    )
    return (resp.choices[0].message.content or "").strip()

# ===================== 推理与执行（显式传参） =====================
def get_compare_by_model_no_history(
    cfg: RunnerConfig,
    client: OpenAI,
    img_path_list: List[str],
    prompt: str,
    model_name: str
) -> str:
    img_b64_list: List[str] = []
    for img_rel in img_path_list or []:
        # img_rel 可以是相对或绝对路径；相对路径基于 cfg.images_root
        abs_path = img_rel if os.path.isabs(img_rel) else os.path.join(cfg.images_root, img_rel)
        with open(abs_path, "rb") as f:
            img_b64_list.append(base64.b64encode(f.read()).decode("utf-8"))

    prompt_base = f"""
    你非常擅长用 React 和 Tailwind 构建互动网页，能精确根据用户提供的多张网页截图还原完整HTML互动网页。

【初始界面要求】
1. 按照用户提供的首张网页截图，构建页面，一定要和给你的首张截图内容完全一致。
2. 不要遗漏任何细节，包括背景色、字体、字号、间距、边框、图标、文本等，都要和截图严格匹配。
3. 截图里的每一句文字都要原样呈现。
4. 图片内容请使用 https://picsum.photos/ 库中的真实图片，url类似https://picsum.photos/id/.../.../...。 **每张图片需显式列出url**，不要使用可复用的图片组件。每个>网页组件的图片url一定要固定，不要用随机数每次重新生成。

【任务要求】
1. 用户会发给你多张图片，每张图片代表网页在经过一个互动操作之后产生变化后的截图，所有图片代表了在本页面上所进行的所有操作后产生的截图。
2. 用户会发给你一个详细的操作序列清单，操作清单中的每一项代表一个操作序列，其中会告诉你起始图（进行操作之前的页面截图）是发给你的第几张图，以及操作序列图（操作序
列中的每一步结束后产生的截图）是发给你的第几张图，请自行找到这些图片。每一个操作序列可能会包含多个操作，也就是涵盖多张图。有一些操作序列包括很多中间过程操作，多>图中只有第一张图和最后一张图，请自行识别具体操作内容。
3. 当你找到这些图片后，请阅读操作清单中本项的操作描述，有三种操作：“输入” “点击” “选择”，请在截图中正确识别互动组件是哪一个，并在生成的html网页中正确地实现他们。
4. 所有给你的互动操作一定都要在生成的html中完美复刻，也就是同时具有完整的功能，且完成后页面与对应截图一致。

请使用以下库： 
<script src="https://cdn.jsdelivr.net/npm/react@18.0.0/umd/react.development.js"></script>
<script src="https://cdn.jsdelivr.net/npm/react-dom@18.0.0/umd/react-dom.development.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@babel/standalone/babel.js"></script>
用以下脚本引入 Tailwind： <script src="https://cdn.tailwindcss.com"></script>
可以用 Google Fonts。
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"></link>

代码输出格式：
只输出完整的 <html></html> 标签内的代码
代码前后不要加 markdown “” 或 “html”

操作清单：{prompt}"""

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_base}]
    for img in img_b64_list:
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}})

    messages = [{"role": "user", "content": content}]
    try:
        return get_response(client, messages, model_name, cfg)
    except Exception as e:
        print(f"[LLM-ERROR] model={model_name}: {e}")
        return ""

# ===================== Worker（显式传参） =====================
def worker(task_queue: Queue, result_queue: Queue, model_name: str, cfg: RunnerConfig):
    client = make_client(cfg)
    while True:
        item = task_queue.get()
        if item is None:
            break
        idx, data = item
        try:
            out_txt = get_compare_by_model_no_history(
                cfg=cfg,
                client=client,
                img_path_list=data.get("image_list"),
                prompt=data.get("prompt"),
                model_name=model_name
            )
            result = {"id": data.get("id"), "output": out_txt}
        except Exception as e:
            result = {"id": data.get("id"), "output": "", "error": str(e)}
        result_queue.put((idx, result))

# ===================== 单模型运行（显式传参） =====================
def run_for_model(cfg: RunnerConfig, model_name: str, rows: List[Dict[str, Any]], output_path: str):
    html_out_dir = resolve_html_dir_for_model(cfg, model_name)

    task_queue: Queue = Queue(maxsize=cfg.num_workers * 8)
    result_queue: Queue = Queue()
    workers: List[Process] = []

    for _ in range(cfg.num_workers):
        p = Process(target=worker, args=(task_queue, result_queue, model_name, cfg))
        p.start()
        workers.append(p)

    for idx, row in enumerate(rows):
        task_queue.put((idx, row))
    for _ in range(cfg.num_workers):
        task_queue.put(None)

    results: List[Any] = [None] * len(rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf8") as fout:
        for _ in tqdm(range(len(rows)), desc=f"{model_name}"):
            idx, result = result_queue.get()
            results[idx] = result
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

            html_code = extract_html_from_any(result)
            if html_code:
                rid = str(result.get("id") if result.get("id") is not None else idx)
                html_path = os.path.join(html_out_dir, f"{rid}.html")
                with open(html_path, "w", encoding="utf-8") as hf:
                    hf.write(html_code)

    print(f"[OK] Output JSONL: {output_path}")
    print(f"[OK] HTML saved in: {html_out_dir}")

    for p in workers:
        p.join()

# ===================== 主函数（显式传参） =====================
def main():
    cfg, _ = load_config_or_exit()
    with open(cfg.input_jsonl, "r", encoding="utf8") as fin:
        rows = [json.loads(line.strip()) for line in fin if line.strip()]

    for model_name in cfg.models:
        output_path = os.path.join(f"{cfg.output_prefix}", f"{model_name}_results.jsonl")
        run_for_model(cfg, model_name, rows, output_path)

if __name__ == "__main__":
    main()
