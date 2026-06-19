"""
export_weights.py
─────────────────
Extracts quantized dense-layer weights from the trained TFLite model
and writes them as C .inc files for inclusion in inference_engine.c

Run AFTER the training notebook:
    cd python_training
    python export_weights.py

Output files (written to riscv_firmware/weights/):
    layer1_weights.inc   layer1_biases.inc
    layer2_weights.inc   layer2_biases.inc
    quant_params.inc     (multipliers & shifts)
"""

import os
import sys
import struct
import numpy as np

TFLITE_PATH  = "./crack_model_quant.tflite"
OUT_DIR      = "../riscv_firmware/weights"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load TFLite model and inspect layers ────────────────────────────────
try:
    import tensorflow as tf
except ImportError:
    sys.exit("ERROR: TensorFlow not installed. Run: pip install tensorflow==2.13.0")

if not os.path.exists(TFLITE_PATH):
    sys.exit(f"ERROR: {TFLITE_PATH} not found. Run the training notebook first.")

# Load interpreter
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

print("=== TFLite Model Tensor Details ===")
tensor_details = interpreter.get_tensor_details()
for t in tensor_details:
    print(f"  [{t['index']:3d}] {t['name']:<60s} {str(t['shape']):<20s} {t['dtype']}")

# ── Find the two Dense layers ────────────────────────────────────────────
# Dense head: typically named 'dense/MatMul' and 'dense_1/MatMul'
dense_weight_tensors = []
dense_bias_tensors   = []

for t in tensor_details:
    name = t['name'].lower()
    if ('dense' in name or 'fully_connected' in name) and 'matmul' not in name:
        continue
    if 'matmul' in name or ('dense' in name and t['dtype'] == np.int8):
        data = interpreter.get_tensor(t['index'])
        if data.ndim == 2:
            dense_weight_tensors.append((t['name'], data, t['quantization']))
    if 'bias' in name and ('dense' in name or 'fully_connected' in name):
        data = interpreter.get_tensor(t['index'])
        if data.ndim == 1:
            dense_bias_tensors.append((t['name'], data, t['quantization']))

if not dense_weight_tensors:
    print("\nWARNING: Could not auto-detect dense layers.")
    print("Scanning all INT8 tensors with 2D shape...")
    for t in tensor_details:
        if t['dtype'] == np.int8:
            data = interpreter.get_tensor(t['index'])
            if data.ndim == 2:
                dense_weight_tensors.append((t['name'], data, t['quantization']))
                print(f"  Found: {t['name']} shape={data.shape}")

if len(dense_weight_tensors) < 2:
    print(f"\nERROR: Expected at least 2 dense weight tensors, found {len(dense_weight_tensors)}")
    print("       Please manually identify tensor indices from the list above")
    print("       and edit this script to use interpreter.get_tensor(<idx>)")
    sys.exit(1)

def write_int8_inc(path, data_flat, values_per_line=16):
    """Write a flat int8 array as a .inc file (comma-separated C literals)."""
    with open(path, 'w') as f:
        chunks = [data_flat[i:i+values_per_line]
                  for i in range(0, len(data_flat), values_per_line)]
        lines = [', '.join(f'{int(v):4d}' for v in chunk) for chunk in chunks]
        f.write(',\n'.join(lines))
        f.write('\n')
    print(f"  Written: {path}  ({len(data_flat)} values)")

def write_int32_inc(path, data_flat, values_per_line=8):
    """Write a flat int32 bias array as a .inc file."""
    with open(path, 'w') as f:
        chunks = [data_flat[i:i+values_per_line]
                  for i in range(0, len(data_flat), values_per_line)]
        lines = [', '.join(f'{int(v):12d}' for v in chunk) for chunk in chunks]
        f.write(',\n'.join(lines))
        f.write('\n')
    print(f"  Written: {path}  ({len(data_flat)} values)")

# ── Export Layer 1 (Dense 64 in → 64 out) ───────────────────────────────
print("\n=== Exporting Layer 1 (Dense head, first layer) ===")
name1, w1, q1 = dense_weight_tensors[0]
print(f"  Name : {name1}")
print(f"  Shape: {w1.shape}  (out_features x in_features)")
print(f"  Quant: scale={q1[0]:.6f}, zero_point={q1[1]}")

write_int8_inc(f"{OUT_DIR}/layer1_weights.inc", w1.flatten())

# Biases for layer 1
if dense_bias_tensors:
    name_b1, b1, qb1 = dense_bias_tensors[0]
    write_int32_inc(f"{OUT_DIR}/layer1_biases.inc", b1.flatten())
else:
    # Generate zero biases as fallback
    zeros = np.zeros(w1.shape[0], dtype=np.int32)
    write_int32_inc(f"{OUT_DIR}/layer1_biases.inc", zeros)
    print("  WARNING: No bias tensor found for Layer 1, using zeros.")

# ── Export Layer 2 (Dense 64 in → 1 out, sigmoid) ───────────────────────
print("\n=== Exporting Layer 2 (Dense head, output layer) ===")
name2, w2, q2 = dense_weight_tensors[1]
print(f"  Name : {name2}")
print(f"  Shape: {w2.shape}  (out_features x in_features)")
print(f"  Quant: scale={q2[0]:.6f}, zero_point={q2[1]}")

write_int8_inc(f"{OUT_DIR}/layer2_weights.inc", w2.flatten())

if len(dense_bias_tensors) >= 2:
    name_b2, b2, qb2 = dense_bias_tensors[1]
    write_int32_inc(f"{OUT_DIR}/layer2_biases.inc", b2.flatten())
else:
    zeros = np.zeros(w2.shape[0], dtype=np.int32)
    write_int32_inc(f"{OUT_DIR}/layer2_biases.inc", zeros)
    print("  WARNING: No bias tensor found for Layer 2, using zeros.")

# ── Compute requantization multipliers ──────────────────────────────────
print("\n=== Computing Requantization Parameters ===")

def compute_quantized_multiplier(input_scale, weight_scale, output_scale):
    """
    Compute (multiplier, shift) for multiply_by_quantized_multiplier().
    Based on TFLite kernel: tensorflow/lite/kernels/internal/quantization_util.cc
    """
    double_multiplier = float(input_scale) * float(weight_scale) / float(output_scale)
    if double_multiplier == 0.0:
        return 0, 0
    # Decompose into mantissa and exponent
    import math
    shift = 0
    mantissa = double_multiplier
    while mantissa < 0.5:
        mantissa *= 2.0
        shift += 1
    while mantissa >= 1.0:
        mantissa /= 2.0
        shift -= 1
    # mantissa now in [0.5, 1.0)
    q30 = round(mantissa * (1 << 30))
    return int(q30), int(shift)

# Approximate output scales from quantization parameters
out1_scale = q1[0] if q1[0] > 0 else 1.0 / 127.0
out2_scale = q2[0] if q2[0] > 0 else 1.0 / 127.0
in_scale   = 1.0 / 255.0  # input image normalized

mult1, shift1 = compute_quantized_multiplier(in_scale,   q1[0], out1_scale)
mult2, shift2 = compute_quantized_multiplier(out1_scale, q2[0], out2_scale)

print(f"  Layer 1: multiplier={mult1}, shift={shift1}")
print(f"  Layer 2: multiplier={mult2}, shift={shift2}")

# Write quant params as a C include
with open(f"{OUT_DIR}/quant_params.inc", 'w') as f:
    f.write(f"/* Auto-generated requantization parameters */\n")
    f.write(f"#define L1_MULTIPLIER  {mult1}\n")
    f.write(f"#define L1_SHIFT       {shift1}\n")
    f.write(f"#define L1_OUT_ZERO    {q1[1]}\n\n")
    f.write(f"#define L2_MULTIPLIER  {mult2}\n")
    f.write(f"#define L2_SHIFT       {shift2}\n")
    f.write(f"#define L2_OUT_ZERO    {q2[1]}\n")
print(f"  Written: {OUT_DIR}/quant_params.inc")

print("\n✅ Weight export complete!")
print("   Now copy these .inc files to riscv_firmware/weights/")
print("   and rebuild in Freedom Studio.")
