import os
import asyncio
import base64
import difflib
import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import psutil  # 需 pip install psutil
import aiofiles
import uvicorn
import sys
from gym import spaces
from playwright.async_api import async_playwright
import argparse

import math
import io
from PIL import Image
from pydantic import BaseModel


class WebHtmlGymEnv:
    metadata = {'render_modes': ['human'], "render_fps": 4}

    def __init__(self, html_file="page.html", viewport_width=1920, viewport_height=1080):
        self.html_file = html_file
        self.is_url = str(html_file).startswith("http://") or str(html_file).startswith("https://")
        self.action_space = spaces.Discrete(3)
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.viewport_height, self.viewport_width, 3), dtype=np.uint8)
        self.p = None
        self.browser = None
        self.context = None
        self.page = None
        self.click_history = []

    async def setup(self):
        # 初始化网页
        self.p = await async_playwright().start()
        self.browser = await self.p.chromium.launch(headless=True)
        self.context = await self.browser.new_context(
            viewport={'width': self.viewport_width, 'height': self.viewport_height},
            device_scale_factor=1
        )
        self.page = await self.context.new_page()

    async def reset(self):
        # 重置网页
        if self.page is None:
            await self.setup()
        await self.page.set_viewport_size({
            "width": self.viewport_width,
            "height": self.viewport_height
        })
        if self.is_url:
            await self.page.goto(self.html_file, timeout=60000)
        else:
            await self.page.goto(f"file:///{os.path.abspath(self.html_file)}", timeout=60000)
        self.click_history = []

    async def load_url(self, url: str):
        self.html_file = url
        self.is_url = True
        if self.page is None:
            await self.setup()
        await self.page.goto(url, timeout=60000)
        self.click_history = []
        print(f"已加载新页面: {url}")

    async def render(self, mode="human"):
        await asyncio.sleep(1)  # 可调
        size = await self.page.evaluate("""() => {
            return {
                width: Math.max(
                    document.documentElement.clientWidth || 1920,
                    document.body ? document.body.scrollWidth : 1920,
                    document.documentElement.scrollWidth || 1920,
                    document.documentElement.offsetWidth || 1920
                ),
                height: Math.max(
                    document.documentElement.clientHeight || 1080,
                    document.body ? document.body.scrollHeight : 1080,
                    document.documentElement.scrollHeight || 1080,
                    document.documentElement.offsetHeight || 1080
                )
            };
        }""")
        await self.page.set_viewport_size({
            "width": max(size["width"], 1920),
            "height": max(size["height"], 1080)
        })
        obs = await self.page.screenshot()
        return obs  # bytes



    async def render_sized(self,
        jpeg_quality=95
    ):
        """
        用于模型输入的：渲染后将宽、高都等比缩小到原来的1/1.3倍，返回bytes。
        """
        obs = await self.render()   # type: bytes
        image = Image.open(io.BytesIO(obs))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        w, h = image.size

        scale = 1/1.3   # 缩小到原来的 1/1.3 倍
        w_bar = int(w * scale)
        h_bar = int(h * scale)

        # resize到目标尺寸
        image = image.resize((w_bar, h_bar), Image.BICUBIC)
        buffered = io.BytesIO()
        image.save(buffered, format='JPEG', quality=jpeg_quality)
        buffered.seek(0)
        # 返回bytes
        return buffered.getvalue()

    async def get_all_dom_elements(self):
        tag_names = [ "input", "button", "textarea", "a", "[role=button]", "[role=checkbox]", "[role=radio]", ]
        doms = []
        for tag in tag_names:
            elems = await self.page.query_selector_all(tag)
            for elem in elems:
                try:
                    props = dict()
                    props['tag'] = await self.page.evaluate('el=>el.tagName.toLowerCase()', elem)
                    props['id'] = await self.page.evaluate('el=>el.id', elem)
                    props['class'] = await self.page.evaluate('el=>el.className', elem)
                    props['name'] = await self.page.evaluate('el=>el.getAttribute("name")', elem)
                    props['type'] = await self.page.evaluate('el=>el.getAttribute("type")', elem)
                    props['placeholder'] = await self.page.evaluate('el=>el.getAttribute("placeholder")', elem)
                    props['text'] = await self.page.evaluate('(el)=>(el.innerText||"")', elem)
                    # 可拓展更多属性
                    doms.append(props)
                except Exception as e:
                    print(f"get_all_dom_elements err: {e}")
        return doms

    
    async def get_dom_tree_with_id(self):
        js_code = """
        () => {
            let nodeId = 0;
            let id2xpath = {};
            function getXPath(node) {
                if (node.nodeType !== 1) return '';
                if (!node.parentNode || node === document.documentElement) return '/' + node.tagName;
                let ix = 1;
                let sib = node.previousSibling;
                while (sib) {
                    if (sib.nodeType === 1 && sib.tagName === node.tagName) ix++;
                    sib = sib.previousSibling;
                }
                return getXPath(node.parentNode) + '/' + node.tagName + '[' + ix + ']';
            }
            function isButtonVisibleAndEnabled(elem) {
                if (!elem) return false;
                try { elem.scrollIntoView({block: 'center', inline: 'center'}); } catch(e) {}
                const style = window.getComputedStyle(elem);
                if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") return false;
                const rect = elem.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return false;
                const cx = rect.left + rect.width / 2;
                const cy = rect.top + rect.height / 2;
                const topElem = document.elementFromPoint(cx, cy);
                let current = topElem;
                while (current) {
                    if (current === elem) return true;
                    current = current.parentElement;
                }
                return false;
            }
            function isInteractiveElement(node) {
                if (!node || !node.tagName) return false;
                const tag = node.tagName.toLowerCase();
                // 典型交互标签
                if (['button', 'a', 'select', 'textarea', 'label'].includes(tag)) return true;
                if (tag === 'input') {
                    const t = (node.getAttribute('type') || 'text').toLowerCase();
                    // 常见可交互 input type
                    return [
                        'button', 'submit', 'reset', 
                        'radio', 'checkbox', 'range', 
                        'file', 'color', 'date', 
                        'time', 'month', 'week', 
                        'email', 'tel', 'password', 'text', 'search', 'number', 'url'
                    ].includes(t);
                }
                return false;
            }
            function isTextInputLike(node) {
                if (!node || !node.tagName) return false;
                const tag = node.tagName.toLowerCase();
                if (tag === 'textarea') return true;
                if (tag === 'input') {
                    const t = (node.getAttribute('type') || 'text').toLowerCase();
                    return (
                        t === 'text' ||
                        t === 'password' ||
                        t === 'search' ||
                        t === 'number' ||
                        t === 'email' ||
                        t === 'tel' ||
                        t === 'url'
                    );
                }
                return false;
            }
            function getVisibleText(node) {
                if (!node || node.nodeType !== 1) return undefined;
                const tag = node.tagName.toLowerCase();
                if (tag === 'input' || tag === 'textarea') {
                    return node.value || '';
                } else if (tag === 'select') {
                    const selected = node.selectedOptions && node.selectedOptions[0];
                    return selected ? selected.textContent : '';
                } else {
                    return node.innerText || '';
                }
            }
            function traverse(node, parents=[]) {
                if (node.tagName && node.tagName.toUpperCase() === 'SCRIPT') {
                    return null;
                }
                // 忽略纯文本节点
                if (node.nodeType !== Node.ELEMENT_NODE) return null;
                const tag = node.tagName ? node.tagName.toLowerCase() : '';
                let can_interact = false;
                let input_value = undefined;
                let visible_text = undefined;
                let should_keep = false;
                let options = undefined; // 新增
                if (isInteractiveElement(node)) {
                    should_keep = true;
                    // 针对 button/checkbox/radio 决定是否可交互
                    try {
                        can_interact = isButtonVisibleAndEnabled(node);
                    } catch (e) {
                        can_interact = false;
                    }
                }
                if (isTextInputLike(node)) {
                    try {
                        input_value = node.value;
                    } catch (e) {
                        input_value = null;
                    }
                }
                if (tag === 'select') {
                    should_keep = true;
                    try {
                        // options提取
                        options = [];
                        for (let opt of node.options) {
                            options.push({
                                value: opt.value,
                                text: opt.text,
                                selected: opt.selected
                            });
                        }
                    } catch (e) { options = null; }
                }
                const new_parents = parents.slice();
                if (node.tagName) new_parents.push(node.tagName);
                const children = [];
                for (let child of node.childNodes) {
                    const childResult = traverse(child, new_parents);
                    if (childResult !== null) {
                        children.push(childResult);
                    }
                }
                if (!should_keep && children.length === 0) return null;
                const id = nodeId++;
                const xpath = getXPath(node);
                id2xpath[id] = xpath;
                const attrs = {};
                if (node.attributes) {
                    for (let attr of node.attributes) {
                        attrs[attr.name] = attr.value;
                    }
                }
                let result = {
                    id: id,
                    tag: node.tagName,
                    attrs: attrs,
                    children: children
                };
                if (should_keep) {
                    result['can_interact'] = can_interact;
                    if (isTextInputLike(node)) {
                        result['input_value'] = input_value;
                    }
                    if (['button', 'input', 'textarea', 'select', 'a', 'label'].includes(tag)) {
                        result['visible_text'] = getVisibleText(node);
                    }
                    if (tag === 'select') {
                        result['options'] = options; // 新增
                    }
                }
                return result;
            }
            const domtree = traverse(document.body, []);
            return {domtree, id2xpath};
        }
        """
        result = await self.page.evaluate(js_code)
        return result["domtree"], result["id2xpath"]


    async def click(self, unique_id: int, id2xpath: dict):
        xpath = id2xpath.get(str(unique_id)) or id2xpath.get(int(unique_id))
        if not xpath:
            print(f"[CLICK] No xpath found for id={unique_id}")
            return False
        elem = await self.page.query_selector(f'xpath={xpath}')
        if elem is None:
            print(f"[CLICK] No element found by xpath {xpath} for id={unique_id}")
            return False
        try:
            await elem.scroll_into_view_if_needed()
            await elem.click()
            self.click_history.append(("click", unique_id))
            return True
        except Exception as e:
            print(f"[CLICK] Failed click: {e}")
            return False

    async def enter(self, unique_id, content, id2xpath):
        xpath = id2xpath.get(str(unique_id)) or id2xpath.get(int(unique_id))
        if not xpath:
            print(f"[ENTER] No xpath found for id={unique_id}")
            return False
        elem = await self.page.query_selector(f'xpath={xpath}')
        if elem is None:
            print(f"[ENTER] No element found by xpath {xpath} for id={unique_id}")
            return False
        try:
            tag = await elem.evaluate("el => el.tagName.toLowerCase()")
            if tag not in ["input", "textarea"]:
                print(f"[ENTER] Element id={unique_id} is {tag}, not input/textarea, cannot enter text.")
                return False

            # ------- 判断是否为数字输入框 ------- #
            is_number_input = await elem.evaluate(
                """el => {
                    if (el.tagName.toLowerCase() === 'input' && el.type && el.type.toLowerCase() === 'number') {
                        return true;
                    }
                    if (el.pattern && el.pattern.match(/^\\d+$/)) { // 正则仅允许数字
                        return true;
                    }
                    if (el.inputMode && el.inputMode.toLowerCase().includes('numeric')) {
                        return true;
                    }
                    // class带有number/num较为常见的情况
                    if ((el.className || '').match(/(number|num)/i)) {
                        return true;
                    }
                    return false;
                }"""
            )

            fill_content = str(content)
            if is_number_input:
                # 尝试转换为 float
                try:
                    # content 为纯数字或数字字符串
                    fill_content = str(float(content))
                except Exception:
                    # 若失败，则尝试提取字符串里的数字（float）
                    m = re.search(r"[-+]?\d*\.?\d+", str(content))
                    if m:
                        fill_content = str(float(m.group()))
                    else:
                        # 都不行就填 0
                        fill_content = "0"

            await elem.fill(fill_content)
            self.click_history.append(("enter", unique_id, fill_content))
            return True
        except Exception as e:
            print(f"[ENTER] Failed to enter text: {e}")
            return False


    async def select(self, unique_id: Any, value: Any, id2xpath: Dict[Any, str]) -> bool:
        """
        根据 unique_id 在 id2xpath 中找到 xpath，定位元素并执行“选择/填写”操作：
        - <select>：优先按 value 精确选择；失败则按选项 text 做最近似匹配并选择
        - <input type="radio">：点击单选
        - <input type="date">：填写日期
        其它类型：不处理

        成功会把动作记录到 self.click_history。
        返回 True/False 表示是否成功。
        """
        # 1) 解析 xpath
        xpath: Optional[str] = id2xpath.get(str(unique_id)) or id2xpath.get(int(unique_id)) if isinstance(unique_id, (str, int)) else None
        if not xpath:
            print(f"[SELECT] No xpath found for id={unique_id}")
            return False

        # 2) 查询元素
        elem = await self.page.query_selector(f"xpath={xpath}")
        if elem is None:
            print(f"[SELECT] No element found by xpath {xpath} for id={unique_id}")
            return False

        try:
            # 3) 判断元素标签类型
            tag = await elem.evaluate("el => el.tagName.toLowerCase()")

            if tag == "select":
                # 3.a) 直接按 value 选择
                try:
                    result = await elem.select_option(str(value))
                    if result and result != []:
                        self.click_history.append(("select", unique_id, value))
                        return True

                    # 3.b) 精确选择失败，获取所有选项并做最近似 text 匹配
                    options = await elem.evaluate(
                        """
                        el => Array.from(el.options).map(o => ({value: o.value, text: o.text}))
                        """
                    )
                    print(f"[SELECT] No exact value '{value}' found; options are: {options}")

                    option_texts = [opt["text"] for opt in options]
                    closest = difflib.get_close_matches(str(value), option_texts, n=1, cutoff=0.0)
                    if not closest:
                        print(f"[SELECT] No similar option (by text) for value: {value}")
                        return False

                    closest_text = closest[0]
                    for opt in options:
                        if opt["text"] == closest_text:
                            print(f"[SELECT] Using closest option: [{opt['value']}] '{opt['text']}' for '{value}'")
                            result = await elem.select_option(opt["value"])
                            if result and result != []:
                                self.click_history.append(("select-closest", unique_id, opt["value"]))
                                return True

                    print("[SELECT] Failed to select even after closest-match logic.")
                    return False

                except Exception as e:
                    print(f"[SELECT] Exception during <select> handling: {e}")
                    return False

            elif tag == "input":
                input_type = await elem.get_attribute("type")

                if input_type == "radio":
                    await elem.scroll_into_view_if_needed()
                    await elem.click()
                    self.click_history.append(("select-radio", unique_id))
                    return True

                elif input_type == "date":
                    await elem.fill(str(value))
                    self.click_history.append(("select-date", unique_id, value))
                    return True

                else:
                    print(f"[SELECT] Element id={unique_id} is input but not type=date/radio, skipping.")
                    return False

            else:
                print(f"[SELECT] Element id={unique_id} is {tag}, not select/input[type=date|radio], cannot select value.")
                return False

        except Exception as e:
            print(f"[SELECT] Failed to select: {e}")
            return False
        
    async def save_state(self):
        return list(self.click_history)

    async def restore_state(self, click_path):
        await self.reset()
        for item in click_path:
            if isinstance(item, int):
                await self.click(item)
            elif isinstance(item, tuple):
                if item[0] == "pixel":
                    _, x, y, button = item
                    await self.click_pixel(x, y, button)
                elif item[0] == "type":
                    _, x, y, text = item
                    await self.type_pixel(x, y, text)
                elif item[0] == "selector":
                    _, selector = item
                    await self.click_by_selector(selector)

    async def save_current_html(self, file_name="my_new_page.html"):
        html = await self.page.evaluate('document.documentElement.outerHTML')
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"{file_name} 保存完成 (from evaluate)")

    async def close(self):
        if self.browser:
            await self.browser.close()
        if self.p:
            await self.p.stop()

# ================== FastAPI service ===================


app = FastAPI()
env: Optional[WebHtmlGymEnv] = None

class ClickRequest(BaseModel):
    id: int
    id2xpath: dict

class EnterRequest(BaseModel):
    id: int
    text: str
    id2xpath: dict

class SelectRequest(BaseModel):
    id: int
    value: str
    id2xpath: dict


class LoadUrlRequest(BaseModel):
    url: str


@app.on_event("startup")
async def startup_event():
    global env
    if len(sys.argv) > 1:
        html_file = sys.argv[1]  # 允许 python server.py xxx.html 这样传参
    else:
        html_file = "page.html"
    env = WebHtmlGymEnv(html_file=html_file)
    await env.reset()
    print(f"WebHtmlGymEnv启动并初始化完成。加载html: {html_file}")

@app.post("/click")
async def api_click(req: ClickRequest):
    global env
    result = await env.click(req.id, req.id2xpath)
    return {"result": result}

@app.post("/enter")
async def api_enter(req: EnterRequest):
    global env
    result = await env.enter(req.id, req.text, req.id2xpath)
    return {"result": result}

@app.post("/select")
async def api_select(req: SelectRequest):
    global env
    result = await env.select(req.id, req.value, req.id2xpath)
    return {"result": result}

@app.get("/observe")
async def api_observe():
    global env
    obs_bytes = await env.render()
    obs_b64 = base64.b64encode(obs_bytes).decode("utf-8")
    return {"image_b64": obs_b64}

@app.get("/observe_sized")
async def api_observe():
    global env
    obs_bytes = await env.render_sized()
    obs_b64 = base64.b64encode(obs_bytes).decode("utf-8")
    return {"image_b64": obs_b64}


@app.get("/observe_sized_small")
async def api_observe():
    global env
    obs_bytes = await env.render_sized_small()
    obs_b64 = base64.b64encode(obs_bytes).decode("utf-8")
    return {"image_b64": obs_b64}


@app.get("/all_dom_elements")
async def api_all_dom_elements():
    global env
    doms = await env.get_all_dom_elements()
    return {"doms": doms}

@app.get("/dom_tree_with_id")
async def api_dom_tree_with_id():
    global env
    domtree, id2xpath = await env.get_dom_tree_with_id()
    return {"domtree": domtree, "id2xpath": id2xpath}

@app.post("/reset")
async def api_reset():
    global env
    await env.reset()
    return {"result": True}

@app.get("/save_state")
async def api_save_state():
    global env
    state = await env.save_state()
    return {"state": state}

@app.post("/restore_state")
async def api_restore_state(request: Request):
    global env
    data = await request.json()
    click_path = data.get("click_path", [])
    await env.restore_state(click_path)
    return {"result": True}

@app.on_event("shutdown")
async def shutdown_event():
    global env
    await env.close()
    print("已关闭WebHtmlGymEnv。")


@app.post("/load_url")
async def api_load_url(req: LoadUrlRequest):
    global env
    await env.load_url(req.url)
    return {"result": True, "message": f"页面已加载: {req.url}"}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("html_or_url", nargs='?', default="page.html")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    # 记录到全局变量，供 event 使用
    startup_html_or_url = args.html_or_url
    uvicorn.run(
        "webenv:app",
        host='0.0.0.0',
        port=args.port,
        reload=True,
        reload_dirs=["./webenv-init"],
    )