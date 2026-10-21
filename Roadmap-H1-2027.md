# H1 2027 — Fine-Tuning & Post-Training Roadmap

**Focus:** MoE architectures, long-horizon agent capabilities, harness-aware context optimization, and emerging multimodal fine-tuning opportunities.

1. **Mixture-of-Experts (MoE) Fine-Tuning:** Evaluate Gemma 4 26B-A4B and NVIDIA Nemotron 3.5 Lightning 30B-A3B for efficient SFT/RL, investigating expert specialization, expert parallelism, and auxiliary-loss-free load balancing where the training recipe permits. (Sources: [Gemma 4](https://deepmind.google/models/gemma/gemma-4/), [Nemotron 3.5](https://research.nvidia.com/labs/nemotron/Nemotron-3.5-Super-and-Lightning/), [DeepSeek-V3](https://arxiv.org/abs/2412.19437))

2. **Long-Horizon Agentic RL & Tool-Calling Fine-Tuning:** Use existing harness and simulation environments to fine-tune multi-turn agents for tool selection, execution, error recovery, replanning, and end-to-end task completion through SFT followed by RL. (Sources: [NVIDIA Nemotron](https://github.com/NVIDIA-NeMo/Nemotron), [Experience-Driven Self-Distillation](https://proceedings.mlr.press/v306/wu26bf.html))

3. **Trajectory-Level Credit Assignment:** Experiment with TRACE-style turn-level reward attribution and process supervision to learn effectively from long tool-use trajectories where final outcome rewards are sparse. (Source: [TRACE](https://arxiv.org/abs/2607.13988))

4. **Synthetic Agent Trajectory Generation:** Build a reusable fine-tuning data pipeline from successful runs, failed tool interactions, and synthetic trajectories, with quality filtering and held-out validation before SFT/RL. (Sources: [Experience-Driven Self-Distillation](https://proceedings.mlr.press/v306/wu26bf.html), [SynthAgent](https://aclanthology.org/2026.acl-long.716/))

5. **Benchmark-Driven Fine-Tuning:** Collaborate with the benchmarking team to turn DeepResearch Bench, HLE, and internal agent failure patterns into targeted training curricula while keeping benchmark test sets out of training data. (Sources: [DeepResearch Bench](https://arxiv.org/abs/2506.11763), [Humanity's Last Exam](https://github.com/centerforaisafety/hle))

6. **Learned Context Compression for Harness:** Explore MEMENTO- and CompactionRL-inspired training to teach models what task-critical state to retain during existing harness compaction, reducing effective context requirements across extended agent sessions. (Sources: [MEMENTO](https://arxiv.org/abs/2604.09852), [CompactionRL](https://arxiv.org/abs/2607.05378))

7. **Multimodal Document Understanding:** Fine-tune vision-language models such as Nemotron 3 Nano Omni 30B-A3B for financial document extraction and reasoning over tables, charts, scanned forms, and related documents. (Sources: [NVIDIA Nano Omni overview](https://developer.nvidia.com/blog/nvidia-nemotron-3-nano-omni-powers-multimodal-agent-reasoning-in-a-single-efficient-open-model/), [NVIDIA fine-tuning recipes](https://docs.nvidia.com/nemo/automodel/latest/model-coverage/omni/nemotron-omni))

8. **Real-Time Speech & Interactive Agent Fine-Tuning:** Investigate Moshi/PersonaPlex-style full-duplex speech models for customer-service training simulations, with emphasis on interruption handling, turn-taking, clarification, and low-latency spoken interaction. (Sources: [PersonaPlex research](https://research.nvidia.com/labs/adlr/personaplex/), [PersonaPlex repository](https://github.com/NVIDIA/personaplex))

## Enterprise relevance

JPMorganChase publicly identifies **speech recognition** as part of its machine-learning expertise and **multimodal document AI** and financial-domain foundation models as research areas; public reporting also describes AI assistance in customer-service call centers. These make document understanding and interactive speech reasonable exploratory fine-tuning use cases, **not evidence that the specific proposed projects have internal approval or existing deployments**. (Sources: [JPMC Machine Learning](https://www.jpmorganchase.com/about/technology/research/machine-learning), [JPMC AI Research](https://www.jpmorganchase.com/about/technology/research/ai), [WSJ reporting](https://www.wsj.com/tech/ai/jpmorgan-chase-artificial-intelligence-banking-939b1b32))
