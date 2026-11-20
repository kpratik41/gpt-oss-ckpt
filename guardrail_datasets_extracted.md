# Dataset Inventory

Extracted from the provided image. `Objects` and `Size (bytes)` are transcribed as displayed.

| Dataset | Objects | Size (bytes) |
|---|---:|---:|
| InjectAgent | 14 | 6,348,488 |
| ATBench | 15 | 35,923,620 |
| Nemotron_Safety_Guard_Dataset_v3 | 38 | 628,428,642 |
| PII_Bench | 4 | 190,252 |
| ExpGuardMix | 4 | 22,422,682 |
| Deepset_Prompt_Injections | 5 | 54,619 |
| NotInject | 18 | 131,720 |
| Microsoft_LLMail_Inject_Challenge | 11 | 2,446,823,536 |
| AgentDojo | 36,707 | 432,847,457 |
| Lakera_Gandalf_Ignore_Instructions | 5 | 54,993 |
| Gretel_Synthetic_Finance_PII_Multilingual | 20 | 153,416,500 |
| OpenAI_Moderation_API_Release | 3 | 480,181 |
| HarmBench | 145 | 79,740,768 |
| NVIDIA_Aegis_AI_Content_Safety_Dataset_2_0 | 7 | 28,458,062 |
| NVIDIA_Nemotron_PII | 4 | 307,308,322 |
| XSTest | 3 | 41,246 |
| OR_Bench | 33 | 33,711,230 |
| PolyGuard | 172 | 124,692,456 |

## S3 Upload Validation ? 2026-11-20

All 18 S3 dataset folders are present. The S3 copies match the source snapshots downloaded during the upload, although several differ from the inventory above because of newer source revisions, different packaging, provenance manifests, and Git metadata.

| Dataset | Expected objects | S3 objects | Expected bytes | S3 bytes |
|---|---:|---:|---:|---:|
| InjectAgent | 14 | 25 | 6,348,488 | 6,429,337 |
| ATBench | 15 | 16 | 35,923,620 | 35,926,257 |
| Nemotron Safety Guard v3 | 38 | 40 | 628,428,642 | 662,518,252 |
| PII Bench | 4 | 5 | 190,252 | 192,713 |
| ExpGuardMix | 4 | 5 | 22,422,682 | 22,425,143 |
| Deepset Prompt Injections | 5 | 6 | 54,619 | 56,884 |
| NotInject | 18 | 9 | 131,720 | 76,103 |
| Microsoft LLMail | 11 | 12 | 2,446,823,536 | 2,446,826,359 |
| AgentDojo | 36,707 | 36,864 | 432,847,457 | 476,218,508 |
| Lakera Gandalf | 5 | 6 | 54,993 | 57,300 |
| Gretel Finance PII | 20 | 21 | 153,416,500 | 153,418,807 |
| OpenAI Moderation | 3 | 7 | 480,181 | 964,635 |
| HarmBench | 145 | 433 | 79,740,768 | 492,345,369 |
| NVIDIA Aegis 2.0 | 7 | 8 | 28,458,062 | 28,460,566 |
| NVIDIA Nemotron PII | 4 | 5 | 307,308,322 | 307,310,783 |
| XSTest | 3 | 4 | 41,246 | 43,707 |
| OR-Bench | 33 | 35 | 33,711,230 | 33,711,324 |
| PolyGuard | 172 | 173 | 124,692,456 | 124,694,917 |
| **Total** |  |  | **4,301,074,774** | **4,791,676,964** |

### Interpretation

- Each S3 folder contains an additional `_source_manifest.json` provenance file.
- AgentDojo, OpenAI Moderation, and HarmBench contain 9 Git metadata objects totaling 151,136,896 bytes.
- Excluding provenance manifests and Git metadata, the source content totals 4,640,528,896 bytes.
- Nemotron Safety Guard v3 includes a newer `adapted_samples.zip` of approximately 34 MB.
- HarmBench's current source is substantially larger; its `data/` directory alone is approximately 368 MB.
- The current NotInject snapshot is complete and contains six Parquet files, `README.md`, and `.gitattributes`.
- AgentDojo and InjecAgent contain more files than the earlier inventory because their current repository revisions have expanded.
- No S3 objects were removed during validation.
