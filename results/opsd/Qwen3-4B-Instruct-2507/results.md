# Qwen3-4B-Instruct-2507 Results

> Bold values indicate the best score for that key among reported values. Missing values are shown as `-`.

## Best-Performing Models

`pass@k` below aggregates the reported `pass@8` and `pass@16` metrics.

### Pass@1

| Model | Metrics Reported | Best Count | Tied Best Count |
| --- | ---: | ---: | ---: |
| Qwen3-4B-Instruct-2507 | 7 | 1 | 0 |
| Qwen3-4B-Instruct-2507_epoch1 | 7 | 0 | 0 |
| Qwen3-4B-Instruct-2507_epoch2 | 7 | 1 | 0 |
| **Qwen3-4B-Instruct-2507_kl_reverse_monte_carlo_epoch1** | 7 | 1 | 2 |
| Qwen3-4B-Instruct-2507_kl_forward_monte_carlo_epoch1_with_correction | 7 | 1 | 0 |
| **Qwen3-4B-Instruct-2507_kl_forward_monte_carlo_epoch1** | 7 | 1 | 2 |

Pass@1 highlight: **Qwen3-4B-Instruct-2507_kl_reverse_monte_carlo_epoch1** and **Qwen3-4B-Instruct-2507_kl_forward_monte_carlo_epoch1** tie for the lead with 1 solo win and 2 tied wins each.

### Pass@K

| Model | Metrics Reported | Best Count | Tied Best Count |
| --- | ---: | ---: | ---: |
| Qwen3-4B-Instruct-2507 | 14 | 4 | 1 |
| Qwen3-4B-Instruct-2507_epoch1 | 0 | 0 | 0 |
| Qwen3-4B-Instruct-2507_epoch2 | 0 | 0 | 0 |
| Qwen3-4B-Instruct-2507_kl_reverse_monte_carlo_epoch1 | 14 | 2 | 1 |
| **Qwen3-4B-Instruct-2507_kl_forward_monte_carlo_epoch1_with_correction** | 14 | 7 | 0 |
| Qwen3-4B-Instruct-2507_kl_forward_monte_carlo_epoch1 | 0 | 0 | 0 |

Pass@K highlight: **Qwen3-4B-Instruct-2507_kl_forward_monte_carlo_epoch1_with_correction** leads with 7 solo wins and 0 tied wins.

## Detailed Results

| Key | Qwen3-4B-Instruct-2507 | Qwen3-4B-Instruct-2507_epoch1 | Qwen3-4B-Instruct-2507_epoch2 | Qwen3-4B-Instruct-2507_kl_reverse_monte_carlo_epoch1 | Qwen3-4B-Instruct-2507_kl_forward_monte_carlo_epoch1_with_correction | Qwen3-4B-Instruct-2507_kl_forward_monte_carlo_epoch1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aime24_pass1_generation_pass_1 | 0.567 | 0.6 | 0.733 | 0.6333333333333333 | **0.7666666666666667** | 0.7 |
| aime24_pass8_generation_pass_8 | **0.8333333333333334** | - | - | **0.8333333333333334** | 0.8 | - |
| aime24_pass16_generation_pass_16 | 0.8333333333333334 | - | - | 0.8333333333333334 | **0.8666666666666667** | - |
| aime25_pass1_generation_pass_1 | 0.467 | 0.533 | 0.433 | **0.5666666666666667** | 0.43333333333333335 | 0.36666666666666664 |
| aime25_pass8_generation_pass_8 | **0.7333333333333333** | - | - | 0.6666666666666666 | 0.7 | - |
| aime25_pass16_generation_pass_16 | **0.7666666666666667** | - | - | 0.6666666666666666 | 0.7333333333333333 | - |
| math500_pass1_generation_pass_1 | 0.904 | 0.914 | **0.916** | 0.9 | 0.908 | 0.91 |
| math500_pass8_generation_pass_8 | 0.948 | - | - | 0.948 | **0.956** | - |
| math500_pass16_generation_pass_16 | 0.954 | - | - | 0.95 | **0.958** | - |
| hmmt25_pass1_generation_pass_1 | 0.333 | 0.333 | 0.333 | **0.3333333333333333** | 0.333333333333333 | **0.3333333333333333** |
| hmmt25_pass8_generation_pass_8 | 0.43333333333333335 | - | - | 0.43333333333333335 | **0.5** | - |
| hmmt25_pass16_generation_pass_16 | 0.5333333333333333 | - | - | 0.4666666666666667 | **0.5666666666666667** | - |
| beyondaime_pass1_generation_pass_1 | **0.39** | 0.35 | 0.32 | 0.34 | 0.34 | 0.33 |
| beyondaime_pass8_generation_pass_8 | 0.57 | - | - | 0.56 | **0.58** | - |
| beyondaime_pass16_generation_pass_16 | 0.63 | - | - | 0.61 | **0.64** | - |
| amobench_pass1_generation_pass_1 | 0.07692307692307693 | 0.10256410256410256 | 0.10256410256410256 | **0.1282051282051282** | 0.10256410256410256 | **0.1282051282051282** |
| amobench_pass8_generation_pass_8 | 0.1794871794871795 | - | - | **0.2564102564102564** | 0.23076923076923078 | - |
| amobench_pass16_generation_pass_16 | 0.23076923076923078 | - | - | **0.358974358974359** | 0.2564102564102564 | - |
| openai/gsm8k_pass1_generation_pass_1 | 0.7672479150871873 | 0.7695223654283548 | 0.7687642153146323 | 0.7717968157695224 | 0.7695223654283548 | **0.775587566338135** |
| openai/gsm8k_pass8_generation_pass_8 | **0.8559514783927218** | - | - | 0.8468536770280516 | 0.8476118271417741 | - |
| openai/gsm8k_pass16_generation_pass_16 | **0.868081880212282** | - | - | 0.8642911296436695 | 0.8627748294162244 | - |
