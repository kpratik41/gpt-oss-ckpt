Weekly update

• Detection now identifies which model produced a text, not just that we produced it. Each served model gets its own key from one escrowed secret, so a hit distinguishes Gemma from Nemotron. Key-to-model bindings are version-controlled and fingerprint-verified, so a wrong secret now fails at startup instead of silently emitting undetectable output.

• Packaged for serving-platform handoff: the inference image installs a five-file wheel with no detector and no HTTP service in it. Three distributions now, so detection runs CPU-only away from the GPU fleet. Integration documented as a vLLM plugin, not a fork — the image stays stock vLLM plus one wheel. Key management and detection are deployable today; the vLLM-side processor still needs writing, ~1–2 weeks and GPU access to validate.
