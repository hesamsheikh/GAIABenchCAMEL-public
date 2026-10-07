# Three task-linked trace cases

Run-record paths below are relative to `data/runs/`. Answers are withheld to avoid republishing GAIA content. The memory banks, `history` transcripts and `_history/` archives named below are not included in this release. `RETRIEVAL_AUDIT.csv` reports inserted character counts. These cases do not identify a causal memory mechanism.

## Procedural overlap: `42576abe-0deb-4869-8c63-225c2d75a95a`

- Baseline row: `gpt_5_6_luna/validation_full_luna_validation_memory_build_v1_attachment_free.jsonl`; score 0; answer withheld.
- Memory row: `gpt_5_6_luna/validation_with_test_memory_full_luna_validation_test_memory_agentic_v1.jsonl`; score 1; answer withheld.
- Selector raw output: `97, 47, 60`.
- Inserted record files, in selector order:

  1. `workflows/gpt_5_6_luna/test_full_luna_test_outcome_blind_agentic_adaptive_memory_build_v1/gpt_5_6_luna_cca708f8277f4d9c920c26e8065b82ad_lang_workflow.md`
  1. `workflows/gpt_5_6_luna/test_full_luna_test_outcome_blind_agentic_adaptive_memory_build_v1/gpt_5_6_luna_66bd1b1c443b4b4ea1080fa06527dd62_ling_workflow.md`
  1. `workflows/gpt_5_6_luna/test_full_luna_test_outcome_blind_agentic_adaptive_memory_build_v1/gpt_5_6_luna_7b10295b528741d7897c663fe194c04d_text_workflow.md`


## Irrelevant-looking retrieval: `11af4e1a-5f45-467d-9aeb-46f4bb0bf034`

- Baseline row: `ministral_3b_2512/validation_full_ministral_validation_baseline_efficiency_v1.jsonl`; score 1; answer withheld.
- Memory row: `ministral_3b_2512/validation_with_test_memory_full_ministral_validation_agentic_memory_toolfree_bank_v1.jsonl`; score 0; answer withheld.
- Selector raw output: `103, 143, 152`.
- Inserted record files, in selector order:

  1. `workflows/ministral_3b_2512/test_full_ministral_test_outcome_blind_agentic_adaptive_memory_build_toolfree_v2/ministral_3b_2512_adfc0afb3ef24e79b66bc49823b6b913_workflow.md`
  1. `workflows/ministral_3b_2512/test_full_ministral_test_outcome_blind_agentic_adaptive_memory_build_toolfree_v2/ministral_3b_2512_f2ea141f073c4a86af4cfd1f2e7903fc_workflow.md`
  1. `workflows/ministral_3b_2512/test_full_ministral_test_outcome_blind_agentic_adaptive_memory_build_toolfree_v2/ministral_3b_2512_fb59de38e68843f9a959a687e6cd0ebc_workflow.md`


## Broad consolidation: `7673d772-ef80-4f0f-a602-1bf4485c9b43`

- Baseline row: `gemini_3_7_flash/validation_full_gemini_validation_baseline_v1.jsonl`; score 1; answer withheld.
- Memory row: `gemini_3_7_flash/validation_with_test_memory_full_gemini_validation_test_memory_agentic_v1.jsonl`; score 0; answer withheld.
- Selector raw output: `2, 9, 13`.
- Inserted record files, in selector order:

  1. `workflows/gemini_3_7_flash/test_full_gemini_test_outcome_blind_agentic_adaptive_memory_build_v1/gemini_3_7_flash_04893fc334fc41178457a717ad01a6a9__workflow.md`
  1. `workflows/gemini_3_7_flash/test_full_gemini_test_outcome_blind_agentic_adaptive_memory_build_v1/gemini_3_7_flash_208b8625c49d4cf195b64ced3bff82fb__workflow.md`
  1. `workflows/gemini_3_7_flash/test_full_gemini_test_outcome_blind_agentic_adaptive_memory_build_v1/gemini_3_7_flash_335024e1aa6a4a48bba5fad4ebd8868c__workflow.md`

For the broad-consolidation case, the selected `gemini_3_7_flash_208b8625c49d4cf195b64ced3bff82fb__workflow.md` has `workflow_version: 25`, 25 task IDs in `gaia_task_names`, and earlier versions in the adjacent `_history/` directory. Its original and later construction tasks can be read in `gemini_3_7_flash/test_full_gemini_test_outcome_blind_agentic_adaptive_memory_build_v1.jsonl`.
