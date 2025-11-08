import json
import os
from pathlib import Path
import argparse
import shutil


def get_start_image_path(file_id):
    return f"./images/{file_id}/start.png"


def map_image_path(orig_path, file_id):
    """将原始 path 映射成统一的 ./images/{file_id}/xxx.png 格式"""
    file_name = os.path.basename(orig_path)
    return f"./images/{file_id}/{file_name}"


def action_seq_to_describe(actions_seq):
    action_map = {
        "input":  lambda a: f'输入-{a["dom_info"].get("visible_text") or a["dom_info"].get("attrs", {}).get("placeholder", "")}-{a.get("input_val")}',
        "select": lambda a: f'选择-{a.get("value")}',
        "click":  lambda a: f'点击-{a["dom_info"].get("visible_text","")}',
    }
    result = []
    for a in actions_seq:
        action = a.get("action")
        if action in action_map:
            try:
                desc = action_map[action](a)
                result.append(desc)
            except Exception:
                result.append(f"{action}-未知")
    return "， ".join(result)


def process_json(in_json, file_id, max_images):
    steps = in_json.get("steps", [])
    prompt_parts = []
    image_list = []
    image_path_to_idx = {}
    compare_idx = 0
    operation_info = []

    def add_image(path):
        std_path = map_image_path(path, file_id)
        if std_path in image_path_to_idx:
            return image_path_to_idx[std_path], std_path
        image_list.append(std_path)
        idx = len(image_list)
        image_path_to_idx[std_path] = idx
        return idx, std_path

    for step in steps:
        if step.get("step_type") != "compare" or not step.get("pass"):
            continue
        compare_idx += 1

        img_before = step.get("img_name_before")
        img_before_path = get_start_image_path(file_id) if img_before == "start" else img_before

        pre_image_list = image_list[:]
        pre_image_path_to_idx = image_path_to_idx.copy()
        pre_prompt_parts = prompt_parts[:]
        pre_operation_info = operation_info[:]

        before_idx, before_path = add_image(img_before_path)
        imgs_after = step.get("img_name_list_after", [])

        if len(imgs_after) == 1:
            after_idx, after_path = add_image(imgs_after[0])
            prompt_type = "单一操作"
            prompt_str = (
                f"第{compare_idx}个互动操作：本操作为{prompt_type}，"
                f"起始图是第{before_idx}张图，操作完成图是第{after_idx}张图，"
                f"操作内部顺序是“{action_seq_to_describe(step.get('actions_seq', []))}”"
            )
            dom_info = step.get('actions_seq', [{}])[0].get('dom_info', {})
            operation_info.append({
                "type": "single",
                "dom_info": dom_info,
                "start_image_path": before_path,
                "end_image_path": after_path
            })
        elif len(imgs_after) >= 2:
            first_idx, first_path = add_image(imgs_after[0])
            last_idx, last_path = add_image(imgs_after[-1])
            prompt_type = "多操作序列"
            prompt_str = (
                f"第{compare_idx}个互动操作：本操作为{prompt_type}，"
                f"起始图是第{before_idx}张图，操作开始图是第{first_idx}张图，"
                f"操作结束图是第{last_idx}张图，"
                f"操作内部顺序是“{action_seq_to_describe(step.get('actions_seq', []))}”"
            )
            dom_info_list = [a.get('dom_info', {}) for a in step.get('actions_seq', [])]
            operation_info.append({
                "type": "multiple",
                "dom_info_list": dom_info_list,
                "start_image_path": before_path,
                "end_image_path": last_path
            })
        else:
            continue

        if len(image_list) > max_images:
            image_list = pre_image_list
            image_path_to_idx = pre_image_path_to_idx
            prompt_parts = pre_prompt_parts
            operation_info = pre_operation_info
            break

        prompt_parts.append(prompt_str)

    if not image_list or not operation_info:
        return None

    prompt_final = "；\n".join(prompt_parts)
    return {
        "id": file_id,
        "prompt": prompt_final,
        "image_list": image_list,
        "operation_info": operation_info
    }


def process_folder(folder_path, out_file_path, max_images):
    # 确保输出路径存在
    out_path = Path(out_file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as outf:
        for file in Path(folder_path).iterdir():
            if file.is_file() and file.suffix == ".json":
                file_id = file.stem
                try:
                    content = file.read_text(encoding="utf-8")
                except Exception as e:
                    print(f"读取文件失败：{file}，错误：{e}")
                    continue

                if not content.strip():
                    print(f"文件为空：{file}")
                    continue

                try:
                    input_json = json.loads(content)
                except Exception as e:
                    print(f"读取json出错：{file}，错误：{e}")
                    continue

                train_item = process_json(input_json, file_id, max_images=max_images)
                if train_item and train_item['image_list']:
                    outf.write(json.dumps(train_item, ensure_ascii=False) + "\n")

    print(f"数据已写入：{out_file_path}")

    # === 新增：拷贝 input_folder ===
    output_parent = out_path.parent
    target_dir = output_parent / "images"
    # 确保目标文件夹存在（若存在则替换）
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(folder_path, target_dir, dirs_exist_ok=True)
    print(f"已拷贝输入文件夹至：{target_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="基于 JSON 配置运行：将 compare 日志转为受限图片数的 JSONL 训练样本")
    parser.add_argument("--config", required=True, help="JSON 配置文件路径")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    folder = cfg.get("input_folder")
    out_file = cfg.get("output_file")
    max_images = cfg.get("max_images", 20)

    errors = []
    if not folder:
        errors.append("缺少必填字段：input_folder")
    if not out_file:
        errors.append("缺少必填字段：output_file")
    if not isinstance(max_images, int) or max_images <= 0:
        errors.append("max_images 必须为正整数")

    if errors:
        raise ValueError("配置错误：\n- " + "\n- ".join(errors))

    return folder, out_file, max_images


if __name__ == "__main__":
    args = parse_args()
    folder_path, out_file, max_images = load_config(args.config)
    process_folder(folder_path, out_file, max_images=max_images)
