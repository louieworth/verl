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

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from recipe.dpo.reference_logps_materializer import _materialize_file, _output_path_for_file


class DummyLocalRunner:
    def compute(self, prompts, responses):
        del prompts, responses
        return np.array([-1.0], dtype=np.float32)


class TestReferenceLogpsMaterializer(unittest.TestCase):
    def _build_config(self, root: Path, allow_cross_namespace_reuse: bool, allow_sample_id_reuse: bool = False):
        return OmegaConf.create(
            {
                "data": {
                    "prompt_key": "prompt",
                    "response_key": "response",
                    "reference_logps_key": "reference_logps",
                    "sample_id_key": "sample_id",
                    "reference_logps_materialized_dir": str(root),
                    "reference_logps_allow_cross_namespace_reuse": allow_cross_namespace_reuse,
                    "reference_logps_allow_sample_id_reuse": allow_sample_id_reuse,
                    "max_prompt_length": 32,
                    "max_response_length": 16,
                    "prompt_truncation": "right",
                    "add_eos": True,
                },
                "actor_rollout_ref": {
                    "model": {
                        "path": "/tmp/model",
                        "tokenizer_path": "/tmp/model",
                        "override_config": {},
                        "trust_remote_code": False,
                    },
                    "ref": {
                        "model": {"path": "/tmp/model"},
                        "fsdp_config": {"model_dtype": "bf16"},
                    },
                    "actor": {
                        "fsdp_config": {"model_dtype": "bf16"},
                    },
                },
            }
        )

    def _write_source_parquet(self, path: Path):
        table = pa.table({"prompt": [["hello"]], "response": ["world"]})
        pq.write_table(table, path)

    def _write_cached_parquet(self, path: Path):
        table = pa.table({"prompt": [["hello"]], "response": ["world"], "reference_logps": [-0.5]})
        pq.write_table(table, path)

    def _write_source_with_sample_ids(self, path: Path, rows: list[tuple[str, str, str]]):
        table = pa.table(
            {
                "prompt": [[prompt] for prompt, _, _ in rows],
                "response": [response for _, response, _ in rows],
                "sample_id": [sample_id for _, _, sample_id in rows],
            }
        )
        pq.write_table(table, path)

    def test_materialize_reuses_cross_namespace_cache_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "train.parquet"
            self._write_source_parquet(source_path)

            config = self._build_config(root, allow_cross_namespace_reuse=True)
            old_namespace = "ac1c8793deadbeef"
            new_namespace = "0a7a5157deadbeef"
            cached_path = Path(
                _output_path_for_file(
                    str(source_path),
                    old_namespace,
                    str(root),
                    explicit_output_name="ref_logps_train",
                )
            )
            self._write_cached_parquet(cached_path)

            reused_path = _materialize_file(
                str(source_path),
                config,
                new_namespace,
                worker_pool=None,
                local_runner=None,
                explicit_output_name="ref_logps_train",
            )

            self.assertEqual(reused_path, str(cached_path))
            new_output_path = _output_path_for_file(
                str(source_path),
                new_namespace,
                str(root),
                explicit_output_name="ref_logps_train",
            )
            self.assertFalse(Path(new_output_path).exists())

    def test_materialize_keeps_exact_namespace_behavior_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "train.parquet"
            self._write_source_parquet(source_path)

            config = self._build_config(root, allow_cross_namespace_reuse=False)
            old_namespace = "ac1c8793deadbeef"
            new_namespace = "0a7a5157deadbeef"
            cached_path = Path(
                _output_path_for_file(
                    str(source_path),
                    old_namespace,
                    str(root),
                    explicit_output_name="ref_logps_train",
                )
            )
            self._write_cached_parquet(cached_path)

            materialized_path = _materialize_file(
                str(source_path),
                config,
                new_namespace,
                worker_pool=None,
                local_runner=DummyLocalRunner(),
                explicit_output_name="ref_logps_train",
            )

            self.assertNotEqual(materialized_path, str(cached_path))
            self.assertTrue(Path(materialized_path).exists())

    def test_materialize_reuses_reference_logps_by_sample_id_for_reordered_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            old_source_path = root / "old_train.parquet"
            new_source_path = root / "new_train.parquet"
            self._write_source_with_sample_ids(
                old_source_path,
                [
                    ("prompt-a", "resp-a", "sample-a"),
                    ("prompt-b", "resp-b", "sample-b"),
                ],
            )
            self._write_source_with_sample_ids(
                new_source_path,
                [
                    ("prompt-b", "resp-b", "sample-b"),
                    ("prompt-a", "resp-a", "sample-a"),
                ],
            )

            config = self._build_config(root, allow_cross_namespace_reuse=False, allow_sample_id_reuse=True)
            old_namespace = "ac1c8793deadbeef"
            new_namespace = "0a7a5157deadbeef"
            cached_path = Path(
                _output_path_for_file(
                    str(old_source_path),
                    old_namespace,
                    str(root),
                    explicit_output_name="ref_logps_train",
                )
            )
            cached_table = pa.table(
                {
                    "prompt": [["prompt-a"], ["prompt-b"]],
                    "response": ["resp-a", "resp-b"],
                    "sample_id": ["sample-a", "sample-b"],
                    "reference_logps": [-0.5, -1.5],
                }
            )
            pq.write_table(cached_table, cached_path)

            materialized_path = _materialize_file(
                str(new_source_path),
                config,
                new_namespace,
                worker_pool=None,
                local_runner=None,
                explicit_output_name="ref_logps_train",
            )

            self.assertNotEqual(materialized_path, str(cached_path))
            output_table = pq.read_table(materialized_path)
            self.assertEqual(output_table.column("sample_id").to_pylist(), ["sample-b", "sample-a"])
            self.assertEqual(output_table.column("reference_logps").to_pylist(), [-1.5, -0.5])


if __name__ == "__main__":
    unittest.main()
