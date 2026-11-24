# MedKAD

Training code for **MedKAD: Knowledge-Aligned On-Policy Distillation for Medical Visual Question Answering**.

The student generates responses from each medical image and question. One frozen teacher scores the same student prefixes under answer-only and answer-plus-medical-knowledge conditions. MedKAD uses positive cosine alignment between the knowledge-induced teacher change and the student learning direction to weight token-level reverse KL. Each response combines uniform and alignment-based weights with mixing coefficient 0.5. Alignment uses the token-ID-deduplicated union of the student and knowledge-conditioned teacher Top-16 candidates; reverse KL uses the student's Top-16 candidates.

## CT training

The launcher uses the paper's default training configuration: eight 80 GB NVIDIA H100 GPUs, 400 steps, batch size 512, eight responses per question, AdamW, constant learning rate 3e-6, 20 warm-up steps, Top-16 candidates, alignment margin 1e-6, and mixing coefficient 0.5. Test accuracy is measured before training and every ten steps; the best test checkpoint is selected. Evaluation samples three responses per question with temperature 0.7, top-p 0.95, and at most 1,024 generated tokens.

```bash
export PYTHON_BIN=/path/to/python
export TRAIN_DATASET=/path/to/ct_train.parquet
export TEST_DATASET_PATH=/path/to/ct_test.parquet
export ACTOR_MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct
export REWARD_MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct
bash train_medkad_ct.sh
```

The training Parquet must include the ground-truth answer, answer text, image, question, and `knowledge_hint`. Set `CUDA_VISIBLE_DEVICES` and `RAY_PORT` if needed. The script writes logs and checkpoints locally. Model weights, datasets, generated medical knowledge, checkpoints, and experiment outputs are distributed separately.

`MedKAD/verl/` contains the training runtime and objective, `MedKAD/opd/` contains medical VQA handling, and `MedKAD/tests/` contains CPU checks. Full GPU training has not been rerun from this cleaned package.
