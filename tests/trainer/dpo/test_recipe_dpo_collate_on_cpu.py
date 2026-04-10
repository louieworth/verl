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

import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from recipe.dpo.collate import pointwise_dynamic_prompt_collate_fn
from recipe.dpo.singlewise_dataset import SingleWiseDPODataset


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(char) % 17 + 2 for char in text]}

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kwargs):
        del tokenize, kwargs
        text = "".join(f"{message['role']}:{message['content']}" for message in messages)
        if add_generation_prompt:
            text += "assistant:"
        return [ord(char) % 17 + 2 for char in text]


class TestRecipeDPOCollate(unittest.TestCase):
    def test_pointwise_collate_trims_shared_prompt_padding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "single_wise.jsonl"
            rows = [
                {"prompt": "ab", "response": "good", "label": 1},
                {"prompt": "abcdef", "response": "ok", "label": 0},
            ]
            data_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            cfg = OmegaConf.create(
                {
                    "max_prompt_length": 8,
                    "max_response_length": 4,
                    "prompt_key": "prompt",
                    "response_key": "response",
                    "label_key": "label",
                    "add_eos": False,
                    "prompt_truncation": "right",
                }
            )

            dataset = SingleWiseDPODataset(str(data_path), tokenizer=DummyTokenizer(), config=cfg)
            batch = pointwise_dynamic_prompt_collate_fn([dataset[0], dataset[1]])

            self.assertEqual(batch["input_ids"].shape, (2, 10))
            self.assertEqual(batch["attention_mask"].shape, (2, 10))
            self.assertEqual(batch["position_ids"].shape, (2, 10))
            self.assertEqual(batch["responses"].shape, (2, 4))
            self.assertEqual(batch["response_mask"].shape, (2, 4))
            self.assertTrue((batch["position_ids"] == batch["attention_mask"].cumsum(dim=-1).clamp_min(1) - 1).all())


if __name__ == "__main__":
    unittest.main()
