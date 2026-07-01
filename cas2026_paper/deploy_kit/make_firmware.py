#!/usr/bin/env python3
"""Generate ready-to-flash firmware folders for the new models by cloning the proven
ESPCN_Micro 32x24->64x48 sketches (ESP32 + Pico) and patching model/arena/op-set.
Pure text; run anywhere. After running, export the INT8 model into each folder:
    python tools/export_tflite.py <arch> <ckpt> <that_folder> --int8 --tile 24 32 --val_dir <calib>
"""
import re, shutil
from pathlib import Path

REPO = Path("/sessions/ecstatic-exciting-mccarthy/mnt/sisr-embedded-systems")
MICRO = REPO/"board_experiments/espcn_micro_thermal_32x24_to_64x48_x2"
ESP_TPL = (MICRO/"esp32_espcn_micro_int8_32x24_to_64x48/esp32_espcn_micro_int8_32x24_to_64x48.ino").read_text()
PICO_TPL = (MICRO/"pico_espcn_micro_int8_32x24_to_64x48/pico_espcn_micro_int8_32x24_to_64x48.ino").read_text()
TEST_IMG = MICRO/"esp32_espcn_micro_int8_32x24_to_64x48/test_image.h"

VARIANTS = {
  "espcn_light": dict(title="ESPCN_Light", ops=["AddConv2D","AddDepthToSpace"],
                      esp32=[64,56,48,40,32], pico=64),          # ReLU fused
  "espcn":       dict(title="ESPCN_Full",  ops=["AddConv2D","AddDepthToSpace","AddTanh"],
                      esp32=[104,96,88,80,72,64], pico=112),      # Tanh not fused
  "fsrcnn":      dict(title="FSRCNN", ops=["AddConv2D","AddTransposeConv","AddPrelu",
                      "AddPad","AddQuantize","AddDequantize"],
                      esp32=[88,80,72,64,56], pico=96),           # onnx2tf op-set: VERIFY (see note)
}

def resolver_block(ops):
    lines = [f"    tflite::MicroMutableOpResolver<{len(ops)}> resolver;"]
    lines += [f"    resolver.{o}();" for o in ops]
    return "\n".join(lines)

def patch(tpl, arch, cfg, board):
    s = tpl
    s = s.replace("espcn_micro_int8_data.h", f"{arch}_int8_data.h")
    s = s.replace("espcn_micro_int8_tflite", f"{arch}_int8_tflite")
    s = s.replace("ESPCN_Micro", cfg["title"])
    # op resolver (matches both esp32/pico: same 3-line block)
    s = re.sub(r"    tflite::MicroMutableOpResolver<2> resolver;\n"
               r"    resolver\.AddConv2D\(\);\n    resolver\.AddDepthToSpace\(\);",
               resolver_block(cfg["ops"]), s)
    if board == "esp32":
        top = cfg["esp32"][0]
        s = s.replace("constexpr int kMaxTensorArenaSize = 64 * 1024;",
                      f"constexpr int kMaxTensorArenaSize = {top} * 1024;")
        arr = "    const size_t candidate_sizes[] = {\n        " + \
              ", ".join(f"{k} * 1024" for k in cfg["esp32"]) + ",\n    };"
        s = re.sub(r"    const size_t candidate_sizes\[\] = \{.*?\};", arr, s, flags=re.S)
    else:
        s = s.replace("constexpr int kTensorArenaSize = 64 * 1024;",
                      f"constexpr int kTensorArenaSize = {cfg['pico']} * 1024;")
    return s

made = []
for arch, cfg in VARIANTS.items():
    base = REPO/f"board_experiments/{arch}_thermal_32x24_to_64x48_x2"
    for board, tpl in (("esp32", ESP_TPL), ("pico", PICO_TPL)):
        d = base/f"{board}_{arch}"
        d.mkdir(parents=True, exist_ok=True)
        (d/f"{board}_{arch}.ino").write_text(patch(tpl, arch, cfg, board))
        shutil.copy(TEST_IMG, d/"test_image.h")
        made.append(str(d.relative_to(REPO)))

print("Generated firmware folders (drop in the exported *_int8_data.h, then compile):")
for m in made: print("  ", m)
print("\nFSRCNN note: its op-set from onnx2tf may differ (PReLU can decompose). If a flash")
print("errors with a missing op, add resolver.Add<Op>() and bump <N>.")
