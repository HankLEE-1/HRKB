"""Stage 2 — Knowledge-augmented Chain-of-Thought generation.

A four-step CoT chain is generated per meme by an MLLM:

  1. visual perception   (c_vis)
  2. context association (c_ctx)  <- retrieved knowledge injected here
  3. mechanism reasoning (c_mech) <- retrieved knowledge injected here
  4. summary induction   (c_sum)

Knowledge is injected only into steps 2-3 (config ``COT.inject_steps``);
injecting into visual perception introduces confirmation bias. Retrieved
prototypes are formatted as a label-agnostic knowledge prefix and placed
inside the context/mechanism instructions.

Each generated chain is stored under ``item["cot_variants"][variant]`` so that
the retrieval-method ablation can reuse the same items with different prefixes.
"""

import os
import re

import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .config import COT
from .retrieve import RetrievalIndex
from .utils import is_chinese, load_json, save_json

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Prompt templates ─────────────────────────────────────────────────────
# Placeholders: {knowledge} is the label-agnostic retrieved prefix. It appears
# only in the Context and Mechanism sections (injection steps 2-3).

_PROMPT_EN = """You are a content-moderation expert analyzing a meme (image + OCR text).
Produce an interpretable, evidence-based chain-of-thought in exactly four sections.
Do NOT output any final verdict or label (e.g. harmful/benign/category).

1. [Visual Perception]: Objectively describe the main entities and the scene in the image.

2. [Context Association]: Decode entities, slang, and cultural references in the text and
   image, and state their surface vs. implied meaning. Use the retrieved rhetorical
   knowledge below as background for decoding:
{knowledge}

3. [Mechanism Reasoning]: Assess *whether* the meme employs any of the retrieved rhetorical
   patterns — consider ironic gaps, metaphor, exaggeration, contrast, or propaganda
   techniques. Phrase findings conditionally; do not presuppose harmfulness.
{knowledge}

4. [Summary Induction]: Concisely summarize the evidence gathered above.

Input text: "{text}"

Output format (strictly):
[Visual Perception]:
[Context Association]:
[Mechanism Reasoning]:
[Summary Induction]:
"""

_PROMPT_CN = """你是一名内容审核专家，正在分析一张梗图（图片 + OCR 文本）。
请生成四段式的、可解释的逐步推理链。不要输出任何最终判定或标签（如 harmful/benign/类别）。

1. [Visual Perception]：客观描述图中的主要实体与场景。

2. [Context Association]：解读文本与图片中的实体、俚语和文化梗，说明其字面义与隐含义。
   请将下面检索到的修辞知识作为解码背景：
{knowledge}

3. [Mechanism Reasoning]：评估该梗图是否采用了检索到的某种修辞模式——考虑反讽落差、隐喻、
   夸张、对比或宣传手法。请用条件式表述，不要预设其有害性。
{knowledge}

4. [Summary Induction]：简明归纳以上证据。

输入文本："{text}"

输出格式（严格遵循）：
[Visual Perception]：
[Context Association]：
[Mechanism Reasoning]：
[Summary Induction]：
"""


def _build_prompt(text: str, knowledge: str, inject_steps) -> str:
    tpl = _PROMPT_CN if is_chinese(text) else _PROMPT_EN
    knowledge = knowledge or "None."
    ctx_know = knowledge if 2 in inject_steps else "None."
    mech_know = knowledge if 3 in inject_steps else "None."
    # Split on the two {knowledge} slots: first = context, second = mechanism.
    parts = tpl.split("{knowledge}")
    if len(parts) == 3:
        prompt = parts[0] + ctx_know + parts[1] + mech_know + parts[2]
    else:
        prompt = tpl.replace("{knowledge}", knowledge)
    return prompt.format(text=text)


# ── Lazy-loaded MLLM ─────────────────────────────────────────────────────

_llm = None
_llm_proc = None


def _get_llm():
    global _llm, _llm_proc
    if _llm is None:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        path = COT["local_model"]
        print(f"Loading local model: {path} ...")
        try:
            _llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                path, torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2", device_map="auto")
        except Exception:
            _llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                path, torch_dtype=torch.bfloat16, device_map="auto")
        _llm_proc = AutoProcessor.from_pretrained(path)
        _llm.eval()
        print("Local model loaded.")
    return _llm, _llm_proc


# ── Parsing ─────────────────────────────────────────────────────────────

_SECTIONS = {
    "visual":   ["[Visual Perception]:", "[Visual Perception]："],
    "context":  ["[Context Association]:", "[Context Association]："],
    "mechanism": ["[Mechanism Reasoning]:", "[Mechanism Reasoning]："],
    "summary":  ["[Summary Induction]:", "[Summary Induction]："],
}


def parse_cot(text: str) -> dict:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    parsed = {k: "" for k in _SECTIONS}
    for key, headers in _SECTIONS.items():
        for hdr in headers:
            m = re.search(re.escape(hdr) + r"(.*?)(?=\n\[[^\]]+\]\s*[::]|$)",
                          text, re.DOTALL)
            if m:
                parsed[key] = m.group(1).strip()
                break
    return parsed


# ── Per-item processing ──────────────────────────────────────────────────

def _process_item(item, image_dir, index, variant, inject_steps) -> bool:
    variants = item.setdefault("cot_variants", {})
    if variant in variants and all(variants[variant].get(k)
                                   for k in _SECTIONS):
        return True

    img_path = os.path.join(image_dir, item["file_name"])
    knowledge = index.retrieve(img_path, item.get("text", ""))
    prompt = _build_prompt(item.get("text", ""), knowledge, inject_steps)

    model, proc = _get_llm()
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text", "text": prompt},
    ]}]
    try:
        from qwen_vl_utils import process_vision_info
        text_in = proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(messages)
        inputs = proc(text=[text_in], images=img_in, videos=vid_in,
                      padding=True, return_tensors="pt").to(_DEVICE)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=1500,
                                 temperature=0.7, top_p=0.9, do_sample=True)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        out = proc.batch_decode(trimmed, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
        raw = out[0] if out else None
    except Exception as e:
        print(f"LLM inference error: {e}")
        return False

    if not raw:
        return False

    parsed = parse_cot(raw)
    if not any(parsed.values()):
        parsed = {"visual": raw, "context": "", "mechanism": "", "summary": ""}
    variants[variant] = {
        "visual": parsed["visual"], "context": parsed["context"],
        "mechanism": parsed["mechanism"], "summary": parsed["summary"],
        "retrieved": knowledge,
    }
    return True


# ── Main pipeline ────────────────────────────────────────────────────────

def generate(data_dir, image_dir, hrkb_path, mode, top_k, variant,
             inject_steps, clip_model, workers=1) -> None:
    index = RetrievalIndex(hrkb_path, mode=mode, top_k=top_k,
                           clip_model=clip_model)

    files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    if not files:
        print(f"No JSON files found in {data_dir}")
        return

    for fname in sorted(files):
        path = os.path.join(data_dir, fname)
        print(f"Processing {fname} ...")
        data = load_json(path)
        pending = [it for it in data
                   if not (it.get("cot_variants", {}).get(variant)
                           and all(it["cot_variants"][variant].get(k)
                                   for k in _SECTIONS))]
        if not pending:
            print(f"  All done for {fname}.")
            continue
        print(f"  {len(pending)} items remaining.")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_process_item, it, image_dir, index,
                                variant, inject_steps): it for it in pending}
            for i, fut in enumerate(tqdm(as_completed(futs), total=len(futs))):
                fut.result()
                if (i + 1) % 10 == 0:
                    save_json(path, data, indent=4)
        save_json(path, data, indent=4)
        print(f"  Saved {fname}.")
