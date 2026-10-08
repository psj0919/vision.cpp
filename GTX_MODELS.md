# GTX model integration

This independent repository is `psj0919/vision.cpp`. `vision-cli` includes:

- `yolo` / `yolov8`: the remote backup's generated YOLOv8n, 640x640 RGB input.
- `resnet` / `resnet18`: generated ResNet18, 224x224 RGB with ImageNet normalization.
- Existing SAM, BiRefNet, depth, Migan and ESRGAN paths.

Both new paths accept `-b cpu` or `-b gtx`. `--log-ops` prints graph operations;
YOLO also supports `--dump-tensor` for raw output comparison. GTX loads the module
at `GGML_BACKEND_PATH` and requires registered device GTX0. Load and device init
errors are reported separately. There is no added CPU fallback scheduler.

Laya runners built by `psj0919/GTX_Compiler` share this library and can load the same
GTX module automatically. GTX is identified separately from other GPU backends so
flash attention defaults off for its graphs. Existing CPU model flags remain intact.

The canonical GGML stays at bundled llama `c66ee8e2c118b9f04a3a500a672f906833cb8875`
with `GGML_MAX_NAME=128`. Build the shared GTX/vision/llama/Laya artifacts using
`gtx_ggml_FPGA/tools/build_unified.sh`. Do not mix incompatible GGML versions.
For full backup restoration, build and run commands see
[compiler migration guide](https://github.com/psj0919/GTX_Compiler/blob/main/MIGRATION.md).

Local CPU validation: YOLO full output equals the old CPU baseline; ResNet matches
its corrected CPU reference; Laya hidden/logits and JSON match the remote CPU baseline.
Actual FPGA execution and full existing-model regressions are unverified.
