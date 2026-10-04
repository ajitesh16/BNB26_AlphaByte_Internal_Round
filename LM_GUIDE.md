# Optional local pretrained-model signal

The frozen `base-v1` diagnosis remains the official benchmark. The local LM is an auxiliary signal and is **not** a failure probability.

## Enable it

From the project folder:

```powershell
pip install -r requirements-lm.txt
$env:BLACKBOX_LM="1"
python lm_check.py data_mock.db
python -m streamlit run app.py
```

The default checkpoint is `sshleifer/tiny-gpt2`, chosen so CPU-only testing is practical. To use another Hugging Face causal language model:

```powershell
$env:BLACKBOX_LM_MODEL="your-model-name"
```

The app will show each step's length-normalized output-surprise z-score and maximum token surprise. These are auxiliary signals; they are not probabilities of failure.

## Ablation

`lm_check.py` evaluates the frozen base model and then `base + local LM` on the same deterministic 70/30 split. Keep the base-v1 result unchanged even if the LM signal does not improve it.
