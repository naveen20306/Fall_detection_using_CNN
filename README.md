# Real-Time Fall Detection System (FPGA Simulation Using Verilog)

This repository contains the design and simulation of a real-time fall detection system based on a Convolutional Neural Network (CNN), implemented entirely in Verilog. The project focuses on simulating the CNN inference for fall detection using grayscale image inputs.

> ⚠️ **Note**: This project is a **simulation-only** design. No physical FPGA implementation was performed.

---

## 🧠 Project Summary

- Simulates a CNN-based fall detection system using Verilog HDL.
- Processes 64x64 grayscale images and classifies them as **"Fall"** or **"Not Fall"**.
- CNN was trained in Python using TensorFlow, and weights were exported as `.mem` files for simulation.
- Convolution, Pooling, Flattening, Dense layers, and Softmax is implemented in Verilog using fixed-point arithmetic.
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
Design code,Testbench code,Python scripts used to train and generate weights and biases and .mem files.

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
- **Simulation**: Xilinx Vivado  

---

## 📌 Status

This project successfully simulates a CNN-based fall detection system in Verilog.

---

## 📬 Contributors
 1.Naveen Kumar B-(naveenau2023@gmail.com)
 2.Sabarish Mohan JS
 3.Hemanth S

---

## 📄 License

[MIT](LICENSE) – Feel free to use and modify with attribution.
