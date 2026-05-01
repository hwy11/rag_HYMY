from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.hymy_rag.cleaning import load_clean_quotes, load_clean_quotes_report
from src.hymy_rag.distill import export_domain_corpora, export_master_corpus
from src.hymy_rag.search import build_index, search_index
from src.hymy_rag.status import build_status_report
from src.hymy_rag.tagging import TaggedImportError, import_tagged, load_tagging_env


ROOT = Path(__file__).resolve().parents[1]


class PipelineTest(unittest.TestCase):
    def test_cleaning_skips_empty_answer_by_default(self) -> None:
        quotes = load_clean_quotes([ROOT / "data" / "raw" / "example.json"])
        self.assertEqual(len(quotes), 2)
        self.assertEqual(quotes[0].date, "2025-11-30")
        self.assertIn("班主任", quotes[0].content)
        self.assertEqual(quotes[1].type_origin, "original_post")

    def test_cleaning_real_schema_maps_answer_and_original_post(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            tmp_path = Path(dirname)
            sample_path = tmp_path / "real.json"
            payload = [
                {
                    "id": "1",
                    "publish_time": "2026-04-28 10:00",
                    "question": "原帖问题很长，足够保留",
                    "answer": None,
                },
                {
                    "id": "2",
                    "publish_time": "2026-04-28 11:00",
                    "question": "这个问题也足够长，能当 trigger",
                    "answer": "短句金句",
                },
            ]
            sample_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            quotes = load_clean_quotes([sample_path], filter_level="none")
            self.assertEqual(len(quotes), 2)
            self.assertEqual(quotes[0].type_origin, "original_post")
            self.assertIsNone(quotes[0].trigger)
            self.assertEqual(quotes[0].content, "原帖问题很长，足够保留")
            self.assertEqual(quotes[1].type_origin, "reply")
            self.assertEqual(quotes[1].trigger, "这个问题也足够长，能当 trigger")
            self.assertEqual(quotes[1].content, "短句金句")

    def test_strict_filter_drops_dual_short_but_keeps_short_reply_with_long_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            tmp_path = Path(dirname)
            sample_path = tmp_path / "real.json"
            payload = [
                {
                    "id": "1",
                    "publish_time": "2026-04-28 10:00",
                    "question": "短问题",
                    "answer": None,
                },
                {
                    "id": "2",
                    "publish_time": "2026-04-28 11:00",
                    "question": "这个触发问题很长，应该保留短回应",
                    "answer": "牛",
                },
            ]
            sample_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            quotes = load_clean_quotes([sample_path], filter_level="strict")
            self.assertEqual(len(quotes), 1)
            self.assertEqual(quotes[0].source_id, "2")

    def test_search_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            tmp_path = Path(dirname)
            tagged_path = tmp_path / "tagged.jsonl"
            index_path = tmp_path / "index.json"
            result = import_tagged([ROOT / "data" / "tagged" / "example_tagged.json"], tagged_path)
            self.assertEqual(result.count, 1)
            build_index(tagged_path, index_path)
            results = search_index(index_path, "工作选择 高考工厂", top_k=5)
            self.assertTrue(results)
            self.assertEqual(results[0]["source_id"], "1")
            filtered = search_index(
                index_path,
                "工作选择 高考工厂",
                top_k=5,
                domains=["职业"],
                quote_types=["方法论"],
                time_sensitivities=["中期"],
            )
            self.assertEqual(len(filtered), 1)
            missing = search_index(
                index_path,
                "工作选择 高考工厂",
                top_k=5,
                domains=["投资"],
            )
            self.assertFalse(missing)

    def test_directory_ingest_and_import(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            tmp_path = Path(dirname)
            raw_dir = tmp_path / "raw"
            tagged_dir = tmp_path / "tagged"
            raw_dir.mkdir()
            tagged_dir.mkdir()
            (raw_dir / "a.json").write_text((ROOT / "data" / "raw" / "example.json").read_text(encoding="utf-8"), encoding="utf-8")
            (tagged_dir / "a.json").write_text((ROOT / "data" / "tagged" / "example_tagged.json").read_text(encoding="utf-8"), encoding="utf-8")

            quotes = load_clean_quotes([raw_dir])
            self.assertEqual(len(quotes), 2)

            tagged_path = tmp_path / "tagged.jsonl"
            result = import_tagged([tagged_dir], tagged_path)
            self.assertEqual(result.count, 1)

    def test_prepare_persona_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            tmp_path = Path(dirname)
            tagged_path = tmp_path / "tagged.jsonl"
            import_tagged([ROOT / "data" / "tagged" / "example_tagged.json"], tagged_path)

            distill_dir = tmp_path / "distill"
            domain_count = export_domain_corpora(tagged_path, distill_dir)
            master_count = export_master_corpus(tagged_path, distill_dir / "_master.md")

            self.assertGreaterEqual(domain_count, 1)
            self.assertEqual(master_count, 1)
            self.assertTrue((distill_dir / "_master.md").exists())

    def test_status_report_guides_next_step(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            tmp_path = Path(dirname)
            raw_dir = tmp_path / "raw"
            processed_path = tmp_path / "processed.jsonl"
            tagged_path = tmp_path / "tagged.jsonl"
            index_path = tmp_path / "index.json"
            persona_dir = tmp_path / "persona"
            distill_dir = tmp_path / "distill"

            raw_dir.mkdir()
            persona_dir.mkdir()
            distill_dir.mkdir()
            (raw_dir / "example.json").write_text((ROOT / "data" / "raw" / "example.json").read_text(encoding="utf-8"), encoding="utf-8")
            index_path.write_text(json.dumps({"docs": []}, ensure_ascii=False), encoding="utf-8")

            report = build_status_report(raw_dir, processed_path, tagged_path, index_path, persona_dir, distill_dir)
            self.assertIn("运行 ingest", report)

    def test_ingest_report_skips_bad_file(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            tmp_path = Path(dirname)
            good = tmp_path / "good.json"
            bad = tmp_path / "bad.json"
            good.write_text((ROOT / "data" / "raw" / "example.json").read_text(encoding="utf-8"), encoding="utf-8")
            bad.write_text("{bad json", encoding="utf-8")
            report = load_clean_quotes_report([good, bad])
            self.assertEqual(report.total_files, 2)
            self.assertEqual(report.successful_files, 1)
            self.assertEqual(len(report.skipped_files), 1)
            self.assertEqual(len(report.quotes), 2)

    def test_near_duplicate_dedup_keeps_longer(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            tmp_path = Path(dirname)
            raw = tmp_path / "dup.json"
            raw.write_text(
                json.dumps(
                    [
                        {"id": 1, "time": "2026-01-01 00:00", "content": "q1", "answer": "平台重要，优先选强平台。"},
                        {"id": 2, "time": "2026-01-01 00:00", "content": "q2", "answer": "平台真的很重要，优先选强平台。"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            quotes = load_clean_quotes([raw], dedup_threshold=0.7)
            self.assertEqual(len(quotes), 1)
            self.assertIn("真的很重要", quotes[0].content)

    def test_import_tagged_rejects_zero_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            tmp_path = Path(dirname)
            empty_dir = tmp_path / "tagged"
            empty_dir.mkdir()
            with self.assertRaises(TaggedImportError):
                import_tagged([empty_dir], tmp_path / "out.jsonl")

    def test_load_tagging_env_reads_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            tmp_path = Path(dirname)
            env_path = tmp_path / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENROUTER_API_BASE=https://openrouter.ai/api/v1",
                        "OPENROUTER_API_KEY=test-key",
                        "OPENROUTER_MODEL=test-model",
                    ]
                ),
                encoding="utf-8",
            )
            env = load_tagging_env(env_path)
            self.assertEqual(env["api_base"], "https://openrouter.ai/api/v1")
            self.assertEqual(env["api_key"], "test-key")
            self.assertEqual(env["model"], "test-model")


if __name__ == "__main__":
    unittest.main()
