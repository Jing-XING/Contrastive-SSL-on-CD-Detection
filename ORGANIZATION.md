# Organization and runtime notes

The local server archive and the existing GitHub default branch contain the same 35 files after normalizing line endings. The GitHub version, including its 19 existing commits, is the integration base; there is no separate unpublished server implementation to overlay.

The original source and archive stay in the numbered local project directory. Changes are limited to baseline/EDCP argument handling and safe entry points, command documentation, and offline CLI checks. The backbone, losses, data splitting, optimizers and evaluation equations are unchanged. Existing paper illustrations and upstream notices remain as already published; no new dataset, patient record, weights, logs or historical experiment outputs are added.

## Environment and entry points

- Baseline/EDCP imports PyTorch, torchvision, NumPy and TensorBoard only when running the model workflow. Checkpoint formats depend on the selected SSL method. ImageNet-pretrained backbones may download weights; `--pretrained_backbone=False` controls the baseline Model helper, while EDCP's separate Resnet_without_fc helper still initializes pretrained weights.
- BYOL uses `byol_main.py` and its YAML file, with PyYAML and training dependencies including its historical Apex integration. Configure `data.image_dir`, backbone and checkpoint paths before running.
- Barlow Twins and MoCo retain their original distributed training entry points and configurations. Their cluster/GPU settings have not been converted into a desktop training workflow.
- No pinned full-training environment was present in the repository. `requirements-check.txt` covers only the isolated CLI checks, not research training. Compatibility with current PyTorch/torchvision APIs and historical checkpoints has not been verified.

## Remaining reproduction issues observed in source

- Both evaluation loops still iterate five folds directly; the `Kfold_num` argument does not select an arbitrary number of folds.
- Training augmentation is assigned to a `Subset.transforms` attribute; its application by the underlying ImageFolder needs review before interpreting experiments.
- The baseline training function assumes at least one scheduled test interval has run before returning its test accuracy; using fewer epochs than that interval needs separate handling.
- EDCP enumerates `resnet18_20.pth.tar` through `resnet18_600.pth.tar` at fixed intervals. Other encoder names or checkpoint layouts require explicit adaptation.
- KNN sampling assumes enough images with class indices 0 and 1 and has an unseeded random split. Patient-level partition provenance cannot be inferred from directory names.

These points are documented, not silently rewritten as new experimental methodology. Full model execution, GPU training, real-data evaluation and paper metrics remain unverified by this pass.
