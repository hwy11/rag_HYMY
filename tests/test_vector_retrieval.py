from __future__ import annotations

import unittest

from src.hymy_rag.retrieval.vector_backend import (
    VectorBackend,
    _combine_quote_recall_scores,
    _point_id_from_key,
)


class VectorRetrievalTest(unittest.TestCase):
    def test_prepare_points_reply_indexes_trigger_channels_only(self) -> None:
        backend = VectorBackend()
        points = backend._prepare_points(
            [
                {
                    "id": "1",
                    "content": "博主回答",
                    "source_question": "用户原问题",
                }
            ]
        )
        fields = {point["payload"]["field"] for point in points}
        texts = {point["text"] for point in points}
        self.assertEqual(fields, {"trigger", "trigger_content"})
        self.assertIn("用户原问题", texts)
        self.assertIn("用户原问题 [SEP] 博主回答", texts)
        self.assertNotIn("博主回答", {text for text in texts if text == "博主回答"})

    def test_prepare_points_original_indexes_content_only(self) -> None:
        backend = VectorBackend()
        points = backend._prepare_points(
            [
                {
                    "id": "2",
                    "content": "幻哥原创",
                    "source_question": "",
                }
            ]
        )
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["payload"]["field"], "content")
        self.assertTrue(points[0]["payload"]["is_original_post"])

    def test_combine_quote_recall_scores_uses_seven_three_for_qa(self) -> None:
        score = _combine_quote_recall_scores({"trigger": 1.0, "trigger_content": 0.5})
        self.assertAlmostEqual(score, 0.7 * 1.0 + 0.3 * 0.5)

    def test_combine_quote_recall_scores_boosts_original_posts(self) -> None:
        score = _combine_quote_recall_scores({"content": 0.8})
        self.assertAlmostEqual(score, 0.8 * 1.25)


    def test_point_id_from_key_is_stable(self) -> None:
        first = _point_id_from_key("processed_data_1-1::trigger")
        second = _point_id_from_key("processed_data_1-1::trigger")
        self.assertEqual(first, second)
        self.assertNotEqual(first, _point_id_from_key("processed_data_1-1::content"))


if __name__ == "__main__":
    unittest.main()
