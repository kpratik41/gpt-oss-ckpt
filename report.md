Weekly update

• Detection now identifies which model produced a text, not just that we produced it. Each served model gets its own key from one escrowed secret, so a hit distinguishes Gemma from Nemotron. Key-to-model bindings are version-controlled and fingerprint-verified, so a wrong secret now fails at startup instead of silently emitting undetectable output.

• Packaged for serving-platform handoff: the inference image installs a five-file wheel with no detector and no HTTP service in it. Three distributions now, so detection runs CPU-only away from the GPU fleet. Integration documented as a vLLM plugin, not a fork — the image stays stock vLLM plus one wheel. Key management and detection are deployable today; the vLLM-side processor still needs writing, ~1–2 weeks and GPU access to validate.


One EFS filesystem, fs-01e4e74b2bc0b0c07 (AI Research Shared Storage), is costing approximately $3,000/day because about 314 TB was moved into EFS Standard after being accessed. Its lifecycle policy has “Transition into Standard: On first access” enabled, which is not AWS’s current recommended default. Please change this setting to None, while retaining the existing 14-day transition to IA policy. This will not delete or modify any data and will prevent future reads from moving large amounts of IA data back into the expensive Standard tier.


AWS console steps
1. Open the Amazon EFS console and select Ohio (us-east-2).
2. Choose File systems.
3. Open:
   - File system ID: fs-01e4e74b2bc0b0c07
   - Name: AI Research Shared Storage
4. In the General section, choose Edit.
5. Find Lifecycle management.
6. Keep Transition into IA set to 14 days since last access.
7. Leave any Transition into Archive setting unchanged.
8. Change Transition into Standard from On first access to None.
9. Choose Save changes.
10. Return to the General section and verify:
    - Transition into IA: 14 days
    - Transition into Standard: None
This change prevents future reads of IA files from promoting those files back to costly Standard storage.
