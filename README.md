# Real-Time Fall Detection System (FPGA Simulation Using Verilog)

This repository contains the design and simulation of a real-time fall detection system based on a Convolutional Neural Network (CNN), implemented entirely in Verilog. The project focuses on simulating the CNN inference for fall detection using grayscale image inputs.

> ⚠️ **Note**: This project is a **simulation-only** design. No physical FPGA implementation was performed.

---

## 🧠 Project Summary

- Simulates a CNN-based fall detection system using Verilog HDL.
- Processes 64x64 grayscale images and classifies them as **"Fall"** or **"Not Fall"**.
- CNN was trained in Python using TensorFlow, and weights were exported as `.mem` files for simulation.
- Full inference pipeline (Convolution, Pooling, Flattening, Dense layers, and Softmax) is implemented in Verilog using fixed-point arithmetic.
- A comprehensive testbench loads inputs and verifies the output classification through simulation.

---

## ✅ Features

- **Hardware-Accurate CNN Simulation**  
  Entire inference logic modeled in Verilog for real-time use-case simulation.

- **Custom CNN Model**  
  Lightweight grayscale CNN trained for binary classification (fall vs not fall) with exported `.mem` files.

- **Fixed-Point Design**  
  Uses fixed-point representation for efficient computation and resource estimation.

- **Verilog Testbench**  
  A complete testbench is included to:
  - Load `.mem` files
  - Stimulate the design with input images
  - Observe internal states and output

---

## 🗂 Repository Contents

```
├── rtl/               # Verilog source files for CNN layers
├── tb/                # Verilog testbench files
├── model_files.zip    # Zipped archive of:
│   ├── Python training scripts
│   ├── .mem files for weights, biases, and input images
│   ├── Original trained model files
│   └── Additional helper scripts
└── README.md
```

> 📦 **Note**: The `.mem` files included in this repo are generated specifically for the trained CNN model used in this project and are not generic.

---

## 🧪 Simulation Flow

1. Train the CNN model using Python (TensorFlow).
2. Export trained weights, biases, and test image data to `.mem` format.
3. Load the `.mem` files in the Verilog testbench.
4. Run the simulation using a tool like ModelSim or Vivado Simulator.
5. Observe classification output and waveform behavior.

---

## ⚙️ Tools Used

- **Language**: Verilog HDL  
- **Training**: Python + TensorFlow  
- **Simulation**: ModelSim / Vivado  
- **Target Platforms (Simulated)**: VEGA, Genesys-2 FPGAs

---

## 📌 Status

This project successfully simulates a CNN-based fall detection system in Verilog. It demonstrates the feasibility of hardware-accelerated AI inference pipelines for real-time health monitoring applications on FPGA platforms, though no physical synthesis or deployment is performed.

---

## 📄 License

This project is intended for academic and research use only.
