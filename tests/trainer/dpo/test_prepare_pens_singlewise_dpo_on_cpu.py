# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from recipe.dpo.data.prepare_pens_singlewise_dpo import (
    build_split,
    compute_floor_quantile,
    load_news_lookup,
    normalize_pos_weight,
)
from recipe.dpo.sample_id import compute_sample_id_from_record


class TestPreparePENSSingleWiseDPO(unittest.TestCase):
    def test_builder_writes_prompt_response_label_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            news_path = root / "news.tsv"
            train_path = root / "train.tsv"
            valid_path = root / "valid.tsv"

            news_path.write_text(
                "\n".join(
                    [
                        "News ID\tCategory\tTopic\tHeadline\tNews body\tTitle entity\tEntity content",
                        "N1\tnews\tpolitics\tOriginal headline 1\tBody 1\t\t",
                        "N2\tsports\tsoccer\tOriginal headline 2\tBody 2\t\t",
                        "N3\tlifestyle\thome\tOriginal headline 3\tBody 3\t\t",
                    ]
                ),
                encoding="utf-8",
            )
            train_path.write_text(
                "\n".join(
                    [
                        "UserID\tClicknewsID\tdwelltime\texposure_time\tpos\tpos_weight\tneg\tneg_weight\tstart\tend\tdwelltime_pos",
                        "U1\tN1 N2\t1 2\t\tN3\t1.0\tN2\t0.5\t\t\t1",
                    ]
                ),
                encoding="utf-8",
            )
            valid_path.write_text(
                "\n".join(
                    [
                        "UserID\tClicknewsID\tdwelltime\texposure_time\tpos\tpos_weight\tneg\tneg_weight\tstart\tend\tdwelltime_pos",
                        "U2\tN2\t1\t\tN1\t1.0\tN3\t0.5\t\t\t1",
                    ]
                ),
                encoding="utf-8",
            )

            news_lookup = load_news_lookup(news_path)
            train_stats = build_split(
                train_path,
                "train",
                root / "out",
                news_lookup,
                shard_size=2,
                write_batch_size=2,
                compression="snappy",
                max_input_rows=-1,
                single_file=False,
                sample_mode="positive_only",
            )
            val_stats = build_split(
                valid_path,
                "val",
                root / "out",
                news_lookup,
                shard_size=2,
                write_batch_size=2,
                compression="snappy",
                max_input_rows=-1,
                single_file=False,
                sample_mode="positive_only",
            )

            self.assertEqual(train_stats["records_written"], 1)
            self.assertEqual(val_stats["records_written"], 1)

            train_files = sorted((root / "out" / "train").glob("*.parquet"))
            self.assertTrue(train_files)

            table = pq.read_table(train_files[0])
            row = table.slice(0, 1).to_pylist()[0]

            self.assertEqual(row["response"], "Original headline 3")
            self.assertEqual(row["label"], 1.0)
            self.assertEqual(row["s_dwell"], 1.0)
            self.assertEqual(row["p_ctr"], 0.0)
            self.assertEqual(row["sample_id"], compute_sample_id_from_record(row))
            self.assertEqual(row["user_id"], "U1")
            self.assertEqual(row["candidate_news_id"], "N3")
            self.assertEqual(row["prompt"][0]["role"], "user")
            self.assertIn("Generate a personalized news headline", row["prompt"][0]["content"])
            self.assertIn("1. Original headline 1", row["prompt"][0]["content"])
            self.assertIn("News body: Body 3", row["prompt"][0]["content"])
            self.assertNotIn("Original headline 3", row["prompt"][0]["content"].split("[Candidate News]")[-1])

    def test_builder_writes_single_file_per_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            news_path = root / "news.tsv"
            train_path = root / "train.tsv"

            news_path.write_text(
                "\n".join(
                    [
                        "News ID\tCategory\tTopic\tHeadline\tNews body\tTitle entity\tEntity content",
                        "N1\tnews\tpolitics\tOriginal headline 1\tBody 1\t\t",
                        "N2\tsports\tsoccer\tOriginal headline 2\tBody 2\t\t",
                    ]
                ),
                encoding="utf-8",
            )
            train_path.write_text(
                "\n".join(
                    [
                        "UserID\tClicknewsID\tdwelltime\texposure_time\tpos\tpos_weight\tneg\tneg_weight\tstart\tend\tdwelltime_pos",
                        "U1\tN1\t1\t\tN2\t1.0\tN1\t0.5\t\t\t1",
                    ]
                ),
                encoding="utf-8",
            )

            news_lookup = load_news_lookup(news_path)
            stats = build_split(
                train_path,
                "train",
                root / "single_out",
                news_lookup,
                shard_size=1,
                write_batch_size=1,
                compression="snappy",
                max_input_rows=-1,
                single_file=True,
                sample_mode="positive_only",
            )

            self.assertEqual(stats["records_written"], 1)
            self.assertEqual(stats["shards_written"], 1)
            self.assertEqual(stats["mode"], "single_file")

            train_file = root / "single_out" / "train.parquet"
            self.assertTrue(train_file.exists())

            table = pq.read_table(train_file)
            self.assertEqual(table.num_rows, 1)
            self.assertIn("sample_id", table.column_names)
            self.assertIn("s_dwell", table.column_names)
            self.assertIn("p_ctr", table.column_names)

    def test_builder_writes_negative_ctr_weight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            news_path = root / "news.tsv"
            train_path = root / "train.tsv"

            news_path.write_text(
                "\n".join(
                    [
                        "News ID\tCategory\tTopic\tHeadline\tNews body\tTitle entity\tEntity content",
                        "N1\tnews\tpolitics\tOriginal headline 1\tBody 1\t\t",
                        "N2\tsports\tsoccer\tOriginal headline 2\tBody 2\t\t",
                    ]
                ),
                encoding="utf-8",
            )
            train_path.write_text(
                "\n".join(
                    [
                        "UserID\tClicknewsID\tdwelltime\texposure_time\tpos\tpos_weight\tneg\tneg_weight\tstart\tend\tdwelltime_pos",
                        "U1\tN1\t1\t\tN2\t1.0\tN1\t1.5\t\t\t1",
                    ]
                ),
                encoding="utf-8",
            )

            news_lookup = load_news_lookup(news_path)
            stats = build_split(
                train_path,
                "train",
                root / "out",
                news_lookup,
                shard_size=1,
                write_batch_size=1,
                compression="snappy",
                max_input_rows=-1,
                single_file=True,
                sample_mode="negative_only",
            )

            self.assertEqual(stats["records_written"], 1)
            self.assertEqual(stats["positive_weight_q90"], 1.0)

            table = pq.read_table(root / "out" / "train.parquet")
            row = table.slice(0, 1).to_pylist()[0]
            self.assertEqual(row["label"], 0.0)
            self.assertEqual(row["s_dwell"], 0.0)
            self.assertEqual(row["p_ctr"], 1.0)

    def test_positive_weight_q90_uses_floor_quantile(self):
        weights = [float(value) for value in range(10)]
        self.assertEqual(compute_floor_quantile(weights, 0.9), 8.0)
        self.assertEqual(normalize_pos_weight(10.0, 8.0), 1.0)
        self.assertEqual(normalize_pos_weight(4.0, 8.0), 0.5)


if __name__ == "__main__":
    unittest.main()
