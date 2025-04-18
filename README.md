# Quantum Generative Adversarial Network (QGAN)

This project implements a **Quantum Generative Adversarial Network (QGAN)** using [Qiskit](https://qiskit.org/), combining a parameterized quantum circuit as the generator and a classical neural network as the discriminator. The goal is to generate data that approximates a bimodal real distribution through adversarial training.

## 🌟 Project Highlights

- 🧠 **Hybrid model**: Quantum generator + classical PyTorch discriminator
- ⚛️ **4-qubit, multi-depth quantum circuit** with entanglement
- 🧪 **Binary cross-entropy loss** to drive adversarial training
- 📉 Tracks KL divergence to monitor convergence
- 📊 Final comparisons of real vs generated distributions
- 📷 Quantum circuit diagram rendered with Matplotlib

## 🧰 Requirements

You’ll need Python 3.10 and the following packages:

```bash
pip install -r requirements_qgan.txt
```

The key dependencies are:

- `qiskit==0.43.1`
- `qiskit-terra==0.24.1`
- `qiskit-aer==0.12.0`
- `torch==2.6.0`
- `matplotlib==3.10.1`
- `numpy==2.2.4`
- `scipy==1.15.2`
- `pylatexenc==2.10`

Make sure you have Python 3.10 installed via `pyenv` or a compatible environment manager.

## 🚀 Running the Program

Once your environment is set up and activated:

```bash
python qgan_manual.py
```

This script will:
- Train a QGAN for 1500 epochs
- Generate and save:
  - `loss_plot.png`: training losses over time
  - `final_output.png`: histogram comparing generated and real data
  - `quantum_generator_circuit.png`: the visual structure of the quantum generator
- Print the final KL divergence between generated and real distributions

## 🧪 Dataset

The real data is synthetically generated from a bimodal Gaussian distribution:

- Half of the samples from `N(-2, 0.5)`
- Half from `N(2, 0.5)`

The generator learns to replicate this distribution over time.

## 📈 Results

Example outputs include:
- A decreasing generator loss (not always smooth!)
- Final generator samples approaching bimodal distribution
- KL divergence that quantifies similarity (ideally approaching ~0.1–1.0)

## 📌 File Structure

```bash
├── qgan_manual.py                # Main training script
├── requirements_qgan.txt        # All required Python packages
├── loss_plot.png                # Training loss curves
├── final_output.png             # Final distribution comparison
├── quantum_generator_circuit.png # Quantum generator circuit diagram
```

## 📚 Learning Outcomes

- Hands-on understanding of quantum-classical hybrid models
- Insight into quantum data encoding, measurement, and parameterization
- Practical challenges of training with probabilistic outputs
- Experience debugging unstable GAN training dynamics

## 📦 Future Improvements

- Add backprop-compatible quantum simulators (e.g. `qiskit-machine-learning`)
- Expand to higher-dimensional data
- Implement noise-aware training for real quantum hardware
---

> Created by [Amy Zhou](https://github.com/amyyzhou)  
