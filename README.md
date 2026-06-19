# 🏗️ EdgeAI-CrackDetect-RISCV

> **Concrete Crack Detection on a RISC-V Microcontroller**  
> An end-to-end Edge AI pipeline: image classification → INT8 quantization → bare-metal C inference on SiFive FE310-G002

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![RISC-V](https://img.shields.io/badge/RISC--V-RV32IMAC-blue)](https://riscv.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)](https://tensorflow.org/)
[![Board](https://img.shields.io/badge/Board-VSDSquadron%20PRO-green)](https://www.vlsisystemdesign.com/)

---

## 📋 Overview

This repository documents a complete **Edge AI project** for detecting cracks in concrete surfaces from images. The project classifies input images into two categories:

- ✅ **Negative** — No crack (surface is intact)
- ⚠️ **Positive** — Crack detected (structural risk)

What makes this project unique is its **deployment target**: the SiFive FE310-G002 RISC-V microcontroller with only **16KB of SRAM**, running without an operating system.

The full workflow — from dataset preprocessing and MobileNetV2 transfer learning, through INT8 post-training quantization, C-array export, and bare-metal inference — is documented here step by step. **You do not need the physical board to follow along**; SiFive Freedom Studio's built-in simulator lets you run and observe inference output on your PC.

This project was built as part of the **VLSI System Design (VSD) RISC-V Edge AI** learning path.

---

## 🎯 Why Crack Detection?

| Property | Details |
|---|---|
| **Real-world impact** | Structural health monitoring of bridges, buildings, roads |
| **Binary classification** | Simple, fast to train (~1–2 hrs on CPU) |
| **Tiny dataset** | ~40K images, downloads in minutes from Kaggle |
| **Edge relevance** | Low-power sensors embedded in structures — no cloud needed |
| **Similar to DR project** | Medical imaging → infrastructure imaging, same pipeline |

---

## 🏗️ Project Architecture

```
Input Image (64×64×3 RGB)
         │
         ▼
┌─────────────────────────┐
│  MobileNetV2 Backbone   │  ← Pre-trained ImageNet weights (frozen)
│  (Depthwise Separable   │
│   Convolutions)         │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Global Average Pooling │  ← Spatial dims → 64-dim vector
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Dense(64, ReLU)        │  ← Custom classification head
│  BatchNorm + Dropout    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Dense(1, Sigmoid)      │  ← Binary output (crack probability)
└────────────┬────────────┘
             │
             ▼
     INT8 Quantization
             │
             ▼
     C Array Export
             │
             ▼
  RISC-V Bare-Metal Inference
  (SiFive FE310-G002, 16KB SRAM)
```

---

## 🧩 The Edge AI Challenge: Fitting a CNN in 16KB SRAM

The FE310-G002 has **16KB of SRAM** and **128Mbit (~16MB) QSPI Flash**. Key strategies:

| Challenge | Solution |
|---|---|
| Model too large for RAM | INT8 quantization → ~4× size reduction; weights stored in Flash |
| No floating-point unit | Integer-only arithmetic in bare-metal C |
| No OS / malloc | Static buffers, ping-pong layer outputs |
| No heavy libraries | Custom 200-line inference engine in pure C |
| CNN backbone too complex | Dense head runs on-chip; backbone optionally via TFLite Micro |

---

## 🛠️ Full Pipeline

```
Python (Colab/Jupyter)                    Freedom Studio (PC)
─────────────────────────                 ───────────────────
Dataset (Kaggle)                          Import Project
    │                                          │
    ▼                                          ▼
Image Preprocessing                       Clean & Build (RV32IMAC)
(resize 64×64, augment)                        │
    │                                          ▼
    ▼                                     Run in Simulator
MobileNetV2 Fine-tuning                   (no board needed!)
    │                                          │
    ▼                                          ▼
INT8 Post-Training Quantization           Observe UART console output
    │                                     (crack / no crack result)
    ▼                                          │
Export → model_params.h/.c                     ▼
         test_image.h                     [Optional] Flash to board
         weights/*.inc
```

---

## 📂 Repository Structure

```
EdgeAI-CrackDetect-RISCV/
│
├── python_training/
│   ├── CrackDetection_Training.ipynb  ← Main Colab notebook (run this first)
│   ├── export_weights.py              ← Exports dense-layer weights as C .inc files
│   └── requirements.txt
│
├── riscv_firmware/
│   ├── main.c                         ← Entry point, UART output, LED signal
│   ├── inference_engine.h/.c          ← INT8 dense-layer inference engine
│   ├── uart_utils.h/.c                ← UART print utilities (sim + hardware)
│   ├── model_params.h/.c              ← [AUTO-GENERATED] Quantized model flatbuffer
│   ├── test_image.h                   ← [AUTO-GENERATED] Test sample as INT8 array
│   ├── Makefile                       ← GNU Make build script
│   └── weights/
│       ├── layer1_weights.inc         ← [AUTO-GENERATED] Dense layer 1 weights
│       ├── layer1_biases.inc          ← [AUTO-GENERATED] Dense layer 1 biases
│       ├── layer2_weights.inc         ← [AUTO-GENERATED] Dense layer 2 weights
│       └── layer2_biases.inc          ← [AUTO-GENERATED] Dense layer 2 biases
│
├── docs/
│   ├── sample_images.png              ← [AUTO-GENERATED] dataset samples
│   ├── training_curves.png            ← [AUTO-GENERATED] accuracy/loss plots
│   ├── confusion_matrix.png           ← [AUTO-GENERATED] evaluation results
│   └── test_sample.png                ← [AUTO-GENERATED] exported test image
│
└── README.md
```

> **[AUTO-GENERATED]** files are created by running the Python notebook. Placeholders are provided so the firmware compiles immediately for simulation.

---

## 🚀 Getting Started

### Part A — Train the Model (Python)

#### Step 1 — Setup

```bash
git clone https://github.com/<Your-Username>/EdgeAI-CrackDetect-RISCV.git
cd EdgeAI-CrackDetect-RISCV/python_training
pip install -r requirements.txt
```

Or open `CrackDetection_Training.ipynb` directly in **Google Colab** (recommended — free GPU):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<Your-Username>/EdgeAI-CrackDetect-RISCV/blob/main/python_training/CrackDetection_Training.ipynb)

#### Step 2 — Download Dataset

1. Go to [Kaggle: Concrete Crack Images for Classification](https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification)
2. Download and extract to `python_training/data/`

Expected structure:
```
python_training/data/
├── Negative/   ← ~20,000 images (no crack)
└── Positive/   ← ~20,000 images (crack)
```

#### Step 3 — Run the Notebook

Run all cells in `CrackDetection_Training.ipynb` in order:

| Cell | What it does |
|---|---|
| Step 0 | Install dependencies |
| Step 1 | Imports & config |
| Step 2 | Load & augment dataset |
| Step 3 | Visualize samples |
| Step 4 | Build MobileNetV2 model |
| Step 5 | Train (~10 epochs, ~1.5 hrs CPU / 15 min GPU) |
| Step 6 | Plot training curves |
| Step 7 | Confusion matrix & classification report |
| Step 8 | **INT8 quantization** → `crack_model_quant.tflite` |
| Step 9 | **Export C arrays** → `model_params.h`, `model_params.c` |
| Step 10 | **Export test image** → `test_image.h`, `weights/*.inc` |

#### Step 4 — Export Weights

After the notebook, run:
```bash
python export_weights.py
```
This extracts the dense-layer INT8 weights into `riscv_firmware/weights/*.inc`.

---

### Part B — Run in Freedom Studio (No Board Needed)

#### Step 1 — Install Freedom Studio

Download from [SiFive Freedom Studio releases](https://github.com/sifive/freedom-studio/releases).  
It includes Eclipse IDE + RISC-V GDB + OpenOCD + the RISC-V toolchain.

#### Step 2 — Import the Project

1. Launch Freedom Studio
2. `File` → `Import...` → `Existing Projects into Workspace`
3. Browse to the `riscv_firmware/` folder → `Finish`

#### Step 3 — Configure Build (Simulation Mode)

The firmware is pre-configured for simulation with `SIMULATION_MODE` defined.  
In this mode:
- `printf` routes to the semihosting console (no real UART needed)
- LED GPIO calls are no-ops
- The inference engine runs fully using the static weight arrays

To verify: right-click project → `Properties` → `C/C++ Build` → `Settings` → `Preprocessor` → confirm `SIMULATION_MODE` is listed.

#### Step 4 — Clean and Build

```
Project → Clean...  (select your project)
Project → Build Project
```

Expected output: `crack_detect.elf` in the Debug/ folder.

#### Step 5 — Run in Simulator

1. `Run` → `Debug Configurations...`
2. Select or create a **SiFive GDB OpenOCD Debugging** config
3. Set the **Executable** path to `crack_detect.elf`
4. Under **Startup**, check `Enable semihosting`
5. Click **Debug**

#### Step 6 — Observe Console Output

In the **Console** tab you should see:

```
================================================
  Concrete Crack Detection — Edge AI (RISC-V)
  SiFive FE310-G002 | VSDSquadron PRO
================================================

[INFO] Model size       : 184320 bytes
[INFO] Input dimensions : 64x64x3
[INFO] Quantization     : INT8

[INFO] Loading test image...
[INFO] Raw INT8 output score : 87

┌─────────────────────────────────┐
│  RESULT: ⚠ CRACK DETECTED       │
└─────────────────────────────────┘

[INFO] True label from test_image.h: CRACK
[PASS] ✓ Prediction matches ground truth!
[INFO] Inference complete. Halting core.
```

---

### Part C — Flash to Real Board (Optional)

If you have the VSDSquadron PRO board:

#### Windows — Install USB Drivers First
1. Download [zadig-2.9.exe](https://zadig.akeo.ie/)
2. Connect the board via USB
3. In Zadig: select the device → choose **WinUSB** → `Install Driver`

#### Flash via OpenOCD
```bash
openocd -f interface/ftdi/vsdsquadron.cfg \
        -f target/sifive.cfg \
        -c "program path/to/crack_detect.elf verify reset exit"
```

#### Observe via Serial Monitor
Connect at **115200 baud** to the board's COM port.  
The inference result will print to terminal, and the onboard LED will blink:
- **2 blinks** → No Crack
- **5 blinks** → Crack Detected

---

## 💻 Hardware & Software

### Target Hardware (for board deployment)
| Component | Details |
|---|---|
| Board | VSDSquadron PRO |
| SoC | SiFive FE310-G002 |
| ISA | RV32IMAC (32-bit) |
| SRAM | 16 KB |
| Flash | 128Mbit QSPI |
| Clock | 16 MHz |

### Development Environment
| Tool | Purpose |
|---|---|
| Python 3.10+ | Training pipeline |
| TensorFlow 2.13 | Model training & quantization |
| Google Colab | Free GPU for training |
| Freedom Studio | RISC-V IDE + Simulator |
| RISC-V GNU Toolchain | Cross-compiler (bundled with Freedom Studio) |
| OpenOCD | Board flashing |

---

## 📊 Expected Results

| Metric | Value |
|---|---|
| Training accuracy | ~97% |
| Validation accuracy | ~96% |
| Quantized model size | ~180 KB |
| INT8 inference accuracy | ~94% |
| Inference time (simulated) | ~2 ms at 16 MHz |

---

## 🔬 How the INT8 Inference Engine Works

The custom C inference engine (`inference_engine.c`) performs all computation in integer arithmetic:

### 1. Quantized Matrix Multiply
```c
for (uint16_t o = 0; o < out_feat; o++) {
    int32_t acc = biases[o];                       // int32 accumulator
    for (uint16_t i = 0; i < in_feat; i++) {
        acc += (int32_t)weights[o*in_feat+i]       // int8 × int8 = int32
             * (int32_t)input[i];
    }
    // Requantize: multiply + right-shift back to int8 range
    output[o] = requantize(acc, multiplier, shift);
}
```

### 2. Requantization (no division needed)
The scale factor is pre-decomposed into a **Q2.30 fixed-point multiplier** and a **right-shift count**:
```
real_output = acc × (input_scale × weight_scale / output_scale)
            ≈ (acc × multiplier) >> shift        ← pure integer ops!
```

### 3. Piecewise-Linear Sigmoid
Sigmoid is approximated by 5 linear segments — accurate to ±2 LSB:
```
x < -64  →  output = -127   (sigmoid ≈ 0)
x in [-64,-16]  → linear ramp up
x in [-16,+16]  → steep linear (main transition)
x in [+16,+64]  → linear ramp up
x > +64  →  output = +127   (sigmoid ≈ 1)
```

---

## 🔧 Extending This Project

| Extension | How |
|---|---|
| Add TFLite Micro (full CNN) | Integrate [tflite-micro](https://github.com/tensorflow/tflite-micro) for full MobileNetV2 inference |
| Camera input | Connect OV7670 camera module to GPIO SPI, capture image → INT8 array |
| Continuous monitoring | Replace `while(1) wfi` with a loop that reads GPIO trigger |
| Alert output | Add buzzer or RF transmitter triggered on crack detection |
| Multi-class (severity) | Change sigmoid → softmax, add 3-4 output classes |

---

## 📜 License

MIT License — see [LICENSE](LICENSE)

---

## ✨ Acknowledgments

- **VLSI System Design (VSD)** — RISC-V Edge AI course structure and guidance
- **SiFive** — FE310-G002 documentation and Freedom Studio toolchain
- **Kaggle / Çağlar Fırat Özgenel** — [Concrete Crack Images dataset](https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification)
- **Google** — MobileNetV2 architecture and TensorFlow Lite quantization tools
- Inspired by [AayusHJainCodely/Risv_Edge_AI](https://github.com/AayusHJainCodely/Risv_Edge_AI) (DR classification)
