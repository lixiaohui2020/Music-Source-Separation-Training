from __future__ import annotations

import re
from dataclasses import dataclass, field

from scripts.paper_digest.arxiv_client import Paper
from scripts.paper_digest.summarizer import detect_topics

_CONTRIBUTION_PATTERNS = [
    r"(?:we\s+)?propose\s+(.{20,280}?)(?:\.|$)",
    r"(?:we\s+)?present\s+(.{20,280}?)(?:\.|$)",
    r"(?:we\s+)?introduce\s+(.{20,280}?)(?:\.|$)",
    r"(?:our\s+)?(?:main\s+)?contributions?\s+(?:are|include)\s+(.{20,320}?)(?:\.|$)",
]

_PROBLEM_PATTERNS = [
    r"(?:this paper|we)\s+(?:address|study|investigate|focus on|tackle)\s+(.{20,200}?)(?:\.|$)",
    r"(?:however|yet|challenge is)\s+(.{20,180}?)(?:\.|$)",
]

_PIPELINE_PATTERNS = [
    r"(?:first|initially),?\s+(.{15,120}?)(?:\.|;|, then)",
    r"then,?\s+(.{15,120}?)(?:\.|;|, (?:next|finally))",
    r"(?:next|subsequently),?\s+(.{15,120}?)(?:\.|;|, finally)",
    r"finally,?\s+(.{15,120}?)(?:\.|$)",
    r"(?:consists of|composed of|comprises)\s+(.{20,200}?)(?:\.|$)",
    r"(?:pipeline|framework|architecture)\s+(?:with|includes?|contains?)\s+(.{20,200}?)(?:\.|$)",
]

_RESULT_PATTERNS = [
    r"(?:achieves?|reaches?|obtains?|attains?)\s+([^.]{10,120}?)(?:\.|$)",
    r"(?:outperforms?|surpasses?|beats?)\s+([^.]{10,120}?)(?:\.|$)",
    r"(?:improves?|reduces?|increases?)\s+(?:by\s+)?([^.]{8,100}?)(?:\.|$)",
    r"(?:state[- ]of[- ]the[- ]art|sota)\s+([^.]{8,120}?)(?:\.|$)",
    r"(?:experimental results (?:show|demonstrate|indicate))\s+([^.]{15,160}?)(?:\.|$)",
    r"(?:on (?:the )?[\w\- ]+(?:dataset|benchmark|corpus))[,\s]+([^.]{10,140}?)(?:\.|$)",
]

_METRIC_PATTERN = re.compile(
    r"(\d+\.?\d*\s*(?:%|dB|db)?\s*(?:SDR|SI-SDR|SI-SDRi|PESQ|STOI|WER|BLEU|MOS|F1|accuracy|relative improvement)[^.]{0,60})",
    flags=re.IGNORECASE,
)

_DATASET_PATTERN = re.compile(
    r"\b(MUSDB(?:18)?|LibriSpeech|VCTK|DNS(?:3)?|VoiceBank|AISHELL|Common Voice|WSJ|LJSpeech|"
    r"AVSpeech|DNS Challenge|CHiME|VoxCeleb|LibriMix|WHAM|FSD50K|AudioSet|ESC-50)\b",
    flags=re.IGNORECASE,
)

_METHOD_KEYWORDS = [
    (r"\btransformer\b", "Transformer"),
    (r"\bdiffusion\b", "Diffusion"),
    (r"\bconformer\b", "Conformer"),
    (r"\bllm|large language model\b", "LLM"),
    (r"\bgan\b", "GAN"),
    (r"\bvae|variational\b", "VAE"),
    (r"\bflow matching\b", "Flow Matching"),
    (r"\battention\b", "Attention"),
    (r"\bencoder[- ]decoder\b", "Encoder-Decoder"),
    (r"\bunet\b", "U-Net"),
    (r"\broformer\b", "RoFormer"),
    (r"\bwhisper\b", "Whisper"),
]


@dataclass
class PaperAnalysis:
    brief_summary: str
    core_idea: str
    pipeline_steps: list[str] = field(default_factory=list)
    experiment_results: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _clean_step(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" ,;.")
    if len(text) > 90:
        text = text[:87] + "..."
    return text[0].upper() + text[1:] if text else text


def _match_patterns(text: str, patterns: list[str]) -> str | None:
    lowered = text.lower()
    for pattern in patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            snippet = match.group(1).strip()
            return snippet[0].upper() + snippet[1:] if snippet else snippet
    return None


def _extract_pipeline_steps(abstract: str, methods: list[str]) -> list[str]:
    steps: list[str] = []
    lowered = abstract.lower()

    for pattern in _PIPELINE_PATTERNS:
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            step = _clean_step(match.group(1))
            if step and step not in steps:
                steps.append(step)
            if len(steps) >= 5:
                return steps[:5]

    if len(steps) >= 2:
        return steps[:5]

    # Generic fallback based on detected methods and task keywords
    task = "音频输入"
    if re.search(r"separation|separate", lowered):
        task = "混合音频输入"
    elif re.search(r"speech recognition|transcri", lowered):
        task = "语音输入"
    elif re.search(r"text-to-speech|synthesis", lowered):
        task = "文本/条件输入"

    method = methods[0] if methods else "神经网络模型"
    training = "特征提取与建模"
    if re.search(r"train", lowered):
        training = "模型训练优化"
    if re.search(r"infer|real[- ]time|online", lowered):
        training = "推理部署"

    output = "输出结果"
    if re.search(r"separation|separate", lowered):
        output = "分离后音轨"
    elif re.search(r"transcri", lowered):
        output = "转录文本"
    elif re.search(r"synthesis|generate", lowered):
        output = "合成音频"

    return [task, method, training, output]


def _extract_experiment_results(abstract: str) -> list[str]:
    results: list[str] = []
    lowered = abstract.lower()

    for pattern in _RESULT_PATTERNS:
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            snippet = _clean_step(match.group(1))
            if snippet and snippet not in results:
                results.append(snippet)
            if len(results) >= 4:
                break
        if len(results) >= 4:
            break

    for match in _METRIC_PATTERN.finditer(abstract):
        snippet = match.group(1).strip()
        entry = f"指标：{snippet}"
        if entry not in results:
            results.append(entry)

    # Prefer full comparative sentences mentioning metrics
    for sentence in _sentences(abstract):
        if re.search(r"\d+\.?\d*\s*(?:%|dB|db)|outperform|state[- ]of[- ]the[- ]art|WER|PESQ|SDR", sentence, re.I):
            cleaned = _clean_step(sentence)
            if cleaned and cleaned not in results:
                results.insert(0, cleaned)
        if len(results) >= 5:
            break

    datasets = sorted({d if d.isupper() else d.title() for d in _DATASET_PATTERN.findall(abstract)})
    if datasets:
        results.append(f"数据集：{', '.join(datasets[:4])}")

    if not results:
        for sentence in _sentences(abstract):
            if re.search(r"experiment|evaluat|result|benchmark|ablation", sentence, flags=re.IGNORECASE):
                results.append(_clean_step(sentence))
            if len(results) >= 3:
                break

    return results[:5]


def _detect_methods(text: str) -> list[str]:
    found: list[str] = []
    lowered = text.lower()
    for pattern, label in _METHOD_KEYWORDS:
        if re.search(pattern, lowered, flags=re.IGNORECASE) and label not in found:
            found.append(label)
    return found[:4]


def analyze_paper(paper: Paper) -> PaperAnalysis:
    abstract = paper.abstract
    sentences = _sentences(abstract)
    topics = detect_topics(paper)
    methods = _detect_methods(f"{paper.title} {abstract}")

    contribution = _match_patterns(abstract, _CONTRIBUTION_PATTERNS)
    problem = _match_patterns(abstract, _PROBLEM_PATTERNS)

    core_parts = []
    if problem:
        core_parts.append(f"问题：{problem.rstrip('.')}。")
    if contribution:
        core_parts.append(f"方法：{contribution.rstrip('.')}。")
    elif len(sentences) >= 2:
        core_parts.append(f"方法：{sentences[1]}")
    else:
        core_parts.append(f"方法：{sentences[0] if sentences else paper.title}")

    if methods:
        core_parts.append(f"关键技术：{' / '.join(methods)}。")

    pipeline_steps = _extract_pipeline_steps(abstract, methods)
    experiment_results = _extract_experiment_results(abstract)

    brief = contribution or (sentences[0] if sentences else paper.title)
    topic_note = f"领域：{' / '.join(topics)}"

    return PaperAnalysis(
        brief_summary=f"{brief}（{topic_note}）",
        core_idea=" ".join(core_parts),
        pipeline_steps=pipeline_steps,
        experiment_results=experiment_results,
        topics=topics,
        methods=methods,
    )
