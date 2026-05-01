import re
from typing import Iterable

from .io_utils import read_json, read_text, write_json, write_text
from .paths import (
    ensure_output_dir,
    enriched_json_path,
    final_markdown_path,
    merged_markdown_path,
    processed_json_path,
)


NOISE_PATTERNS = [
    r"^[呲牙流泪发怒睡调皮]+$",
    r"^美得我想哭",
    r"^风景不错",
    r"^吃了顿好的",
    r"^今天天气不咋样",
    r"^刚吃完.*套餐",
    r"^这就是.*的生活",
    r"^有没有必要提一嘴",
]

ENRICHMENT_RULES = [
    {
        "patterns": [
            "女人",
            "女",
            "男",
            "恋爱",
            "结婚",
            "彩礼",
            "小床",
            "伴侣",
            "老婆",
            "老公",
            "出轨",
            "分手",
            "舔狗",
            "女权",
            "打拳",
            "处女",
            "生娃",
            "孩子",
            "家政",
        ],
        "tags": ["#GenderDynamics", "#Relationships", "#Marriage", "#SocialMores"],
        "desc": "Discusses dynamics between men and women, marriage, and social relationships.",
    },
    {
        "patterns": ["钱", "赚", "富", "穷", "资产", "发财", "利润", "成本", "亏", "赢", "收入", "薪资", "工资", "消费", "买"],
        "tags": ["#WealthCreation", "#FinancialLogic", "#MoneyMindset", "#EconomicReality"],
        "desc": "Insights on wealth accumulation, financial logic, and economic realities.",
    },
    {
        "patterns": ["股", "A股", "银行", "房", "买房", "房价", "跌", "涨", "投资", "抄底", "梭哈", "券商", "基金", "市场"],
        "tags": ["#InvestmentStrategy", "#RealEstate", "#StockMarket", "#MarketTrends"],
        "desc": "Analysis of investment strategies, real estate, and market trends.",
    },
    {
        "patterns": ["学", "考", "研", "校", "工作", "业", "职", "公务员", "编制", "老师", "医生", "律师", "专业", "大学", "毕业"],
        "tags": ["#CareerAdvice", "#EducationSystem", "#JobMarket", "#ProfessionalDevelopment"],
        "desc": "Advice on career choices, education paths, and professional development.",
    },
    {
        "patterns": ["肥", "瘦", "吃", "饿", "肉", "餐", "饮", "健康", "身体", "病", "医", "药", "死", "活", "运动", "锻炼", "减肥"],
        "tags": ["#HealthAndWellness", "#DietaryHabits", "#PhysicalHealth", "#Lifestyle"],
        "desc": "Reflections on health, diet, weight loss, and physical well-being.",
    },
    {
        "patterns": ["国", "中", "外", "美", "日", "韩", "欧", "润", "移民", "签证", "世界", "社会", "底层", "阶层", "局势", "政治"],
        "tags": ["#Geopolitics", "#SocialHierarchy", "#InternationalRelations", "#SocietalObservations"],
        "desc": "Observations on geopolitics, social hierarchy, and societal trends.",
    },
]

DEFAULT_TAGS = ["#LifeInsights", "#PersonalThoughts", "#DailyObservations"]
DEFAULT_DESC = "General observations and personal thoughts on daily life."


def iter_batches(start: int = 1, end: int = 10) -> Iterable[int]:
    return range(start, end + 1)


def process_entry(entry_id: str, content: str):
    answer_match = re.search(r"\*\*回答\*\*:(.*)", content, re.DOTALL)
    if answer_match:
        question_part = content[: answer_match.start()].strip()
        answer_part = answer_match.group(1).strip()
    else:
        question_part = content.strip()
        answer_part = None

    time_match = re.search(r"\*\*发布时间\*\*:\s*(.*)", question_part)
    if time_match:
        publish_time = time_match.group(1).strip()
        question_text = re.sub(r"\*\*发布时间\*\*:\s*.*", "", question_part).strip()
    else:
        publish_time = "Unknown"
        question_text = question_part

    is_noise = False
    if not answer_part and len(question_text) < 50:
        is_noise = any(re.search(pattern, question_text) for pattern in NOISE_PATTERNS)

    if not answer_part and len(question_text) < 100 and any(
        keyword in question_text for keyword in ["好看", "震撼", "风景", "天气", "吃完", "电影"]
    ):
        is_noise = True

    if is_noise:
        return None

    return {
        "id": entry_id,
        "publish_time": publish_time,
        "question": question_text,
        "answer": answer_part,
    }


def process_batch(batch_num: int) -> int:
    input_path = merged_markdown_path(batch_num)
    if not input_path.exists():
        return 0

    ensure_output_dir()
    content = read_text(input_path)
    entries = re.split(r"---\n\n##\s+", content)[1:]

    processed_data = []
    for entry in entries:
        id_match = re.match(r"(\d+)\.\n\n(.*)", entry, re.DOTALL)
        if not id_match:
            continue
        result = process_entry(id_match.group(1), id_match.group(2))
        if result:
            processed_data.append(result)

    write_json(processed_json_path(batch_num), processed_data, indent=2)
    return len(processed_data)


def generate_enrichment(text: str):
    if not text:
        return "No content provided.", []

    found_tags = set()
    found_descs = set()
    for rule in ENRICHMENT_RULES:
        if any(pattern in text for pattern in rule["patterns"]):
            found_tags.update(rule["tags"])
            found_descs.add(rule["desc"])

    if not found_tags:
        found_tags.update(DEFAULT_TAGS)
        found_descs.add(DEFAULT_DESC)

    clean_text = text.replace("\n", " ").strip()
    topic_snippet = f"{clean_text[:40]}..." if len(clean_text) > 40 else clean_text
    summary_text = " ".join(found_descs)
    summary = f"{summary_text} Specifically regarding: {topic_snippet}"
    keywords = list(found_tags)[:10]
    return summary, keywords


def enrich_batch(batch_num: int) -> int:
    input_path = processed_json_path(batch_num)
    if not input_path.exists():
        return 0

    ensure_output_dir()
    data = read_json(input_path)
    enriched_data = []
    for entry in data:
        summary, keywords = generate_enrichment(entry.get("question"))
        enriched_data.append(
            {
                "id": entry.get("id"),
                "publish_time": entry.get("publish_time"),
                "question": entry.get("question"),
                "answer": entry.get("answer"),
                "summary": summary,
                "keywords": keywords,
            }
        )

    write_json(enriched_json_path(batch_num), enriched_data, indent=4)
    return len(enriched_data)


def reconstruct_batch(batch_num: int) -> int:
    enriched_path = enriched_json_path(batch_num)
    processed_path = processed_json_path(batch_num)
    input_path = enriched_path if enriched_path.exists() else processed_path
    if not input_path.exists():
        return 0

    ensure_output_dir()
    json_data = read_json(input_path)
    lines = [
        f"# hymy - 第{batch_num}批 (整理后)\n",
        f"**共 {len(json_data)} 条内容**\n\n",
        "---\n\n",
    ]
    for entry in json_data:
        lines.append(f"## {entry['id']}.\n\n")
        lines.append(f"**发布时间**: {entry['publish_time']}\n\n")
        if "summary" in entry:
            lines.append(f"> **概述**: {entry['summary']}\n\n")
        if "keywords" in entry:
            lines.append(f"> **关键字**: {' '.join(entry['keywords'])}\n\n")
        lines.append(f"{entry['question']}\n\n")
        if entry.get("answer"):
            lines.append(f"**回答**:\n\n{entry['answer']}\n\n")
        lines.append("---\n\n")

    write_text(final_markdown_path(batch_num), "".join(lines))
    return len(json_data)


def verify_batch(batch_num: int):
    input_path = final_markdown_path(batch_num)
    if not input_path.exists():
        return None

    lines = read_text(input_path).splitlines()
    entry_count = sum(1 for line in lines if line.startswith("## "))
    answer_tag_count = sum(1 for line in lines if "**回答**:" in line)
    return entry_count, answer_tag_count
