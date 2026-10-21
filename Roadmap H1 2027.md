# H1 2027 — Fine-Tuning & Post-Training Roadmap

**Focus:** MoE architectures, long-horizon agent capabilities, harness-aware context optimization, and emerging multimodal fine-tuning opportunities.

1. **Mixture-of-Experts (MoE) Fine-Tuning:** Evaluate Gemma 4 26B-A4B and NVIDIA Nemotron 3.5 Lightning 30B-A3B for efficient SFT/RL, investigating expert specialization, expert parallelism, and auxiliary-loss-free load balancing where the training recipe permits. [1–3]

2. **Long-Horizon Agentic RL & Tool-Calling Fine-Tuning:** Use existing harness and simulation environments to fine-tune multi-turn agents for tool selection, execution, error recovery, replanning, and end-to-end task completion through SFT followed by RL. [4–5]

3. **Trajectory-Level Credit Assignment:** Experiment with TRACE-style turn-level reward attribution and process supervision to learn effectively from long tool-use trajectories where final outcome rewards are sparse. [6]

4. **Synthetic Agent Trajectory Generation:** Build a reusable fine-tuning data pipeline from successful runs, failed tool interactions, and synthetic trajectories, with quality filtering and held-out validation before SFT/RL. [5, 7]

5. **Benchmark-Driven Fine-Tuning:** Collaborate with the benchmarking team to turn DeepResearch Bench, HLE, and internal agent failure patterns into targeted training curricula while keeping benchmark test sets out of training data. [8–9]

6. **Learned Context Compression for Harness:** Explore MEMENTO- and CompactionRL-inspired training to teach models what task-critical state to retain during existing harness compaction, reducing effective context requirements across extended agent sessions. [10–11]

7. **Multimodal Document Understanding:** Fine-tune vision-language models such as Nemotron 3 Nano Omni 30B-A3B for financial document extraction and reasoning over tables, charts, scanned forms, and related documents. [12–13]

8. **Real-Time Speech & Interactive Agent Fine-Tuning:** Investigate Moshi/PersonaPlex-style full-duplex speech models for customer-service training simulations, with emphasis on interruption handling, turn-taking, clarification, and low-latency spoken interaction. [14–15]

## Enterprise relevance

JPMorganChase publicly identifies **speech recognition** as part of its machine-learning expertise and **multimodal document AI** and financial-domain foundation models as research areas; public reporting also describes AI assistance in customer-service call centers. These make document understanding and interactive speech reasonable exploratory fine-tuning use cases, **not evidence that the specific proposed projects have internal approval or existing deployments**. [16–18]

## References

1. Google DeepMind, *Gemma 4*: https://deepmind.google/models/gemma/gemma-4/
2. NVIDIA Research, *Nemotron 3.5 Super and Lightning* (October 14, 2026): https://research.nvidia.com/labs/nemotron/Nemotron-3.5-Super-and-Lightning/
3. DeepSeek-AI, *DeepSeek-V3 Technical Report* (auxiliary-loss-free MoE load balancing): https://arxiv.org/abs/2412.19437
4. NVIDIA NeMo, *Nemotron Developer Repository* (agentic training recipes): https://github.com/NVIDIA-NeMo/Nemotron
5. Wu et al., *From Interactions to Principles: Experience-Driven Self-Distillation for Evolving LLM Agents* (ICML 2026): https://proceedings.mlr.press/v306/wu26bf.html
6. Tao et al., *TRACE: Turn-level Reward Assignment via Credit Estimation for Long-Horizon Agents* (2026): https://arxiv.org/abs/2607.13988
7. *SynthAgent: Adapting Web Agents with Synthetic Supervision* (ACL 2026): https://aclanthology.org/2026.acl-long.716/
8. Du et al., *DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents*: https://arxiv.org/abs/2506.11763
9. Center for AI Safety, *Humanity's Last Exam* (benchmark and contamination guidance): https://github.com/centerforaisafety/hle
10. Kontonis et al., *MEMENTO: Teaching LLMs to Manage Their Own Context* (2026): https://arxiv.org/abs/2604.09852
11. Li et al., *CompactionRL: Reinforcement Learning with Context Compaction for Long-Horizon Agents* (2026): https://arxiv.org/abs/2607.05378
12. NVIDIA, *Nemotron 3 Nano Omni* (model overview): https://developer.nvidia.com/blog/nvidia-nemotron-3-nano-omni-powers-multimodal-agent-reasoning-in-a-single-efficient-open-model/
13. NVIDIA NeMo AutoModel, *Nemotron-3-Nano-Omni Fine-Tuning Recipes*: https://docs.nvidia.com/nemo/automodel/latest/model-coverage/omni/nemotron-omni
14. NVIDIA Research, *PersonaPlex: Natural Conversational AI With Any Role and Voice*: https://research.nvidia.com/labs/adlr/personaplex/
15. NVIDIA, *PersonaPlex Code and Model Information*: https://github.com/NVIDIA/personaplex
16. JPMorganChase, *Machine Learning Center of Excellence*: https://www.jpmorganchase.com/about/technology/research/machine-learning
17. JPMorganChase, *Artificial Intelligence Research*: https://www.jpmorganchase.com/about/technology/research/ai
18. *The Wall Street Journal*, *The Rise of Artificial Intelligence at JPMorgan* (February 2025; call-center agent assistance): https://www.wsj.com/tech/ai/jpmorgan-chase-artificial-intelligence-banking-939b1b32
