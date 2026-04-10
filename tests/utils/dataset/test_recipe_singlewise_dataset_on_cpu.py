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


class TestRecipeSingleWiseDataset(unittest.TestCase):
    def test_dataset_shapes_and_masks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "single_wise.jsonl"
            row = {
                "prompt": [{"role": "user", "content": "hello world"}],
                "response": "good answer",
                "label": 1,
                "data_source": "unit_test",
            }
            data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            cfg = OmegaConf.create(
                {
                    "max_prompt_length": 8,
                    "max_response_length": 6,
                    "prompt_key": "prompt",
                    "response_key": "response",
                    "label_key": "label",
                    "add_eos": True,
                    "prompt_truncation": "right",
                }
            )

            dataset = SingleWiseDPODataset(str(data_path), tokenizer=DummyTokenizer(), config=cfg)
            sample = dataset[0]

            self.assertEqual(sample["input_ids"].shape[0], 14)
            self.assertEqual(sample["responses"].shape[0], 6)
            self.assertEqual(sample["response_mask"].sum().item(), 6)
            self.assertTrue((sample["labels"][:8] == -100).all())
            self.assertEqual(sample["label"].item(), 1.0)
            self.assertEqual(sample["data_source"], "unit_test")


if __name__ == "__main__":
    unittest.main()
