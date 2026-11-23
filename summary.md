
We set up a repeatable GPU benchmark for Shieldstral, Granite Guardian and both Gemma 4 models. We used Aegis to test broad harm detection, XSTest to measure false alarms on safe prompts, and OpenAI Moderation to score individual safety categories.

We recommended Shieldstral as the default for text content safety. It had the highest F1 across all four OpenAI Moderation categories and falsely flagged only 5 of 250 safe XSTest prompts.

Gemma E4B outperformed E2B on every comparison. In a replay with the same number of false flags, Shieldstral plus E4B caught three more violence cases than Shieldstral alone with an adjusted threshold. We recommended testing E4B as a second reviewer in shadow mode.
