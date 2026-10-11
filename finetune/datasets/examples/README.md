# Dataset fixtures

Machine-generated pipeline fixtures. **Not training data.**

| File | Purpose |
|---|---|
| `synthetic.jsonl` | arithmetic / string-ops SFT records with *computed* answers, so a low loss is meaningful for the pipeline and says nothing about agriculture |
| `placeholder_agriculture.json` | demonstrates the metadata schema only; text is bracketed `[PLACEHOLDER]` and asserts no agronomic facts |
| `manifest.json` | provenance for the synthetic fixture (CC0-1.0, marked `synthetic: true`) |

Regenerate:

```bash
uv run lfm-finetune init-synthetic --destination datasets/examples --examples 12
```

Never train on these. See `../README.md` for the authoring, provenance and
licensing rules.
