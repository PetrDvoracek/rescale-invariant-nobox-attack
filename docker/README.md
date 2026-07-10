# docker/

Reproducible environment for the rescale-invariant no-box attack repo.

`requirements.txt` holds the curated `==` pins for the direct dependencies;
`requirements.lock` is the full transitive `pip freeze` of the working,
checkpoint-producing environment. The pins in `requirements.txt` are derived
from that lock, not from a fresh resolve. **`albumentations` and `timm` are the
two fragile, number-moving pins** (PROGRAM.md §11): augmentation defaults change
the F_Q labels / attack strength, and timm model-name / pretrained-cfg changes
shift reported victim accuracies — re-verify both on any bump. Rebuild with
`docker build -f docker/Dockerfile -t rina .` from the repo root.
