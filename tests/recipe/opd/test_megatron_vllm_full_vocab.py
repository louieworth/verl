# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import torch


def test_vllm_full_vocab_sample_logprobs_stay_vocab_ordered():
    from vllm.v1.sample.sampler import Sampler

    import recipe.gkd.megatron.teacher.vllm_engine  # noqa: F401

    logprobs = torch.log_softmax(torch.randn(3, 7), dim=-1)
    sampled_token_ids = torch.tensor([1, 2, 3], dtype=torch.int64)

    output = Sampler.gather_logprobs(logprobs, -1, sampled_token_ids)

    assert output.logprobs.shape == logprobs.shape
    assert output.logprob_token_ids.shape == (3, 0)
    assert torch.allclose(output.logprobs, logprobs)
