# Local changes to the vendored serl_launcher

Vendored from [HIL-SERL / SERL](https://github.com/rail-berkeley/hil-serl)
([Apache-2.0](LICENSE)), with these changes:

- **Trimmed to what SPARR uses:** the SAC agent and its networks, encoders,
  replay buffers and data stores. The BC and hybrid-arm SAC agents, reward
  classifiers, gym wrappers, evaluation/logging/wandb helpers and robot-specific
  utilities were removed; `utils/launcher.py` keeps only `make_sac_pixel_agent`
  and its augmentation helper.
- **Lazy optional imports:** `utils/train_utils.py` imports `requests` and `tqdm`
  inside the ResNet download helper, so the Phase 2 image does not need them.
- **Dependencies:** pinned in `docker/phase2/source-lock-phase2.env`; this
  directory's `setup.py` is not used (the code is placed on `PYTHONPATH`).
- **Dependency compatibility:** `common/common.py`, `common/encoding.py`, and
  `common/typing.py` contain local compatibility changes.

See [Third-party notices](../THIRD_PARTY_NOTICES.md) for license texts and
attributions.
