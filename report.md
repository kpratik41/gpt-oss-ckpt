Weekly update

• Detection now identifies which model produced a text, not just that we produced it. Each served model gets its own key from one escrowed secret, so a hit distinguishes Gemma from Nemotron. Key-to-model bindings are version-controlled and fingerprint-verified, so a wrong secret now fails at startup instead of silently emitting undetectable output.

• Packaged for serving-platform handoff: the inference image installs a five-file wheel with no detector and no HTTP service in it. Three distributions now, so detection runs CPU-only away from the GPU fleet. Integration documented as a vLLM plugin, not a fork — the image stays stock vLLM plus one wheel. Key management and detection are deployable today; the vLLM-side processor still needs writing, ~1–2 weeks and GPU access to validate.


Corrected the BIRD evaluation harness configuration that had skipped 173 test questions because of an overly restrictive prompt-length limit, then worked with the BIRD team to rerun only the affected questions using the unchanged model checkpoint and inference logic. The repaired evaluation increased ReToolSQL’s test execution accuracy from 71.72% to 78.14%, positioning it at #2 on the BIRD single-model leaderboard.


cat apex-0.1-cp312-cp312-linux_x86_64.whl.part1 apex-0.1-cp312-cp312-linux_x86_64.whl.part2 > apex-0.1-cp312-cp312-linux_x86_64.whl


Yes—designing a deep research agent is a strong ML system design interview question, especially for senior candidates. It can reveal how they reason about retrieval, planning, model behavior, evaluation, and production constraints.
The key is to give it a concrete user, expected output, and operating limits. “Design a deep research agent” alone is broad enough that candidates may spend most of the interview guessing the requirements.
Below are eight questions you can use, with five components candidates should cover for each.
1. Design a deep research agent
“Design an agent that takes a business research question, searches public websites and internal documents, and produces a report with citations. It should handle conflicting sources and finish within 10 minutes under a fixed per-report budget. How would you build and evaluate it?”

Candidates should discuss:
- Planning and orchestration: Breaking the question into subquestions, choosing tools, tracking progress, and deciding when to stop researching.
- Retrieval and source quality: Query generation, search, document parsing, freshness, credibility, and deduplication.
- Evidence and synthesis: Connecting claims to supporting passages, resolving conflicting evidence, citation correctness, and expressing uncertainty.
- Evaluation: Research completeness, factual accuracy, citation support, and usefulness; evaluating individual steps as well as the final report.
- Production controls: Latency and cost budgets, parallel execution, caching, tool failures, document permissions, and prompt injection from retrieved content.
Useful follow-up: “The report reads well, but several citations do not support its claims. How would you diagnose and fix that?”
2. Design a natural-language-to-SQL system
“Design an assistant that lets business analysts ask questions in natural language against a warehouse with hundreds of tables. It should generate and execute read-only SQL and explain the results. How would you make it reliable?”

Candidates should discuss:
- Schema and business context: Retrieving relevant tables, join relationships, metric definitions, and example queries.
- Ambiguity handling: Clarifying terms such as ‘active customer,’ resolving time periods, and preserving conversation context.
- SQL generation and validation: Dialect support, syntax checking, schema checking, query planning, and bounded repair using execution feedback.
- Safe execution: User permissions, tenant isolation, read-only access, query cost limits, and sensitive data handling.
- Evaluation: Semantic correctness and result correctness, difficult joins, ambiguous questions, and tests that avoid relying only on exact SQL matches.
Useful follow-up: “The SQL executes successfully but answers the wrong business question. How would you detect that?”
3. Design an evaluation framework for LLM agents
“Multiple teams are shipping agents that use search, SQL, and internal APIs. Design a shared evaluation platform that helps teams compare versions and decide whether a release is safe to deploy.”

Candidates should discuss:
- Evaluation datasets: Representative tasks, production-derived examples, edge cases, held-out sets, and dataset versioning.
- Evaluation criteria: Final-answer quality, task completion, tool-use correctness, permissions, cost, and latency.
- Judging methods: Deterministic checks, execution-based evaluation, human review, and calibrated model judges.
- Reproducible execution: Trace capture, controlled tool environments, repeated trials, and model/prompt/tool version tracking.
- Release decisions and monitoring: Regression thresholds, uncertainty in scores, slice-level performance, online experiments, and production feedback.
Useful follow-up: “Your model judge prefers longer answers, but users do not. How would you identify and correct that bias?”
4. Design an enterprise knowledge assistant
“Design an assistant that answers employee questions using internal documents. Documents change frequently, may contradict one another, and have different access permissions. Answers must include citations.”

Candidates should discuss:
- Ingestion and indexing: Parsing, chunking, metadata, incremental updates, and deletion propagation.
- Retrieval: Hybrid search, query rewriting, reranking, and selecting context within a token budget.
- Answer generation: Grounding, handling conflicting or stale documents, citations, and abstaining when evidence is insufficient.
- Access control: Permission-aware retrieval, tenant isolation, and preventing restricted information from appearing in answers or caches.
- Evaluation and operations: Separating retrieval failures from generation failures, measuring freshness and answer quality, and managing latency and cost.
5. Design an agent that investigates business metric changes
“Design an agent that investigates questions such as ‘Why did conversion drop last week?’ It can query a warehouse and inspect experiment results and deployment logs. It should return supported hypotheses and recommended next checks.”

Candidates should discuss:
- Problem definition: Metric definitions, comparison periods, seasonality, data freshness, and whether the change is meaningful.
- Investigation planning: Decomposing the metric, choosing segments, generating hypotheses, and prioritizing analyses.
- Tool execution: SQL generation, statistical analysis, joining evidence across systems, and recovering from tool errors.
- Reasoning quality: Distinguishing correlation from causation, accounting for confounders, and avoiding unsupported explanations.
- Evaluation: Known-incident replay, correctness of calculations, evidence quality, investigation efficiency, and expert review.
6. Design a customer support agent that can take actions
“Design a support agent that answers policy questions, checks order status, and issues eligible refunds through APIs. It should complete routine cases and escalate others to a human.”

Candidates should discuss:
- Task routing and context: Identifying intent, retrieving policies, authenticating users, and maintaining conversation state.
- Tool and workflow design: Structured API calls, eligibility checks, state transitions, and separating proposed actions from committed actions.
- Action reliability: Idempotency, retries, partial failures, duplicate refunds, and reconciliation.
- Escalation and boundaries: Approval requirements, uncertain cases, policy exceptions, and useful handoffs to humans.
- Evaluation and monitoring: Resolution quality, incorrect actions, escalation quality, customer outcomes, and audit trails.
7. Design a document extraction system
“Design a system that extracts structured fields and line items from invoices and contracts with varied layouts. It should flag uncertain results for human review and integrate with downstream business systems.”

Candidates should discuss:
- Document processing: OCR, layout understanding, tables, long documents, and poor-quality scans.
- Extraction approach: Model selection, schema-constrained output, field relationships, and normalization.
- Validation: Arithmetic checks, cross-field consistency, evidence locations, and handling missing information.
- Human review: Confidence calibration, review thresholds, correction workflows, and learning from feedback.
- Evaluation and deployment: Field-level and document-level accuracy, generalization to new layouts, error costs, throughput, and privacy.
8. Design conversational analytics over SQL results
“Design an analytics assistant that supports follow-up questions, generates charts, and explains query results. Users should be able to refine an analysis without restating all assumptions.”

Candidates should discuss:
- Conversation state: Tracking filters, metrics, time ranges, assumptions, and changes in user intent.
- Analysis planning: Choosing when to query again, reuse results, ask a clarification, or perform additional calculations.
- Result interpretation: Grounding explanations in computed values, handling empty results, and avoiding unsupported conclusions.
- Visualization: Selecting appropriate charts, preserving units and definitions, and making the analysis reproducible.
- Evaluation: Multi-turn correctness, consistency across SQL, charts, and prose, plus usability, latency, and cost.
