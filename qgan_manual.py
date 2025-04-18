import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import entropy
from qiskit import Aer, QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.visualization import circuit_drawer

# Real data: bimodal distribution
def generate_real_data(n=1000):
    return np.concatenate([
        np.random.normal(-2, 0.5, n // 2),
        np.random.normal(2, 0.5, n // 2)
    ])

real_data = generate_real_data()

# Generator: 4 qubits, 2 layers, entanglement
NUM_QUBITS = 4
DEPTH = 2
params = ParameterVector("θ", length=NUM_QUBITS * 2 * DEPTH)

def create_generator(params):
    qc = QuantumCircuit(NUM_QUBITS)
    offset = 0
    for d in range(DEPTH):
        for i in range(NUM_QUBITS):
            qc.ry(params[offset + i], i)
            qc.rx(params[offset + i + NUM_QUBITS], i)
        offset += NUM_QUBITS * 2
        for i in range(NUM_QUBITS - 1):
            qc.cx(i, i + 1)
    qc.measure_all()
    return qc

def sample_from_generator(params_tensor, num_samples=1000):
    qc = create_generator(params_tensor)
    backend = Aer.get_backend("qasm_simulator")
    job = backend.run(qc, shots=num_samples)
    result = job.result()
    counts = result.get_counts()

    samples = []
    for bitstring, count in counts.items():
        decimal = int(bitstring, 2)
        scaled = (decimal / (2**NUM_QUBITS - 1)) * 6 - 3
        samples += [scaled] * count
    return np.array(samples).reshape(-1, 1)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

params_tensor = torch.tensor(
    np.random.uniform(0, 2 * np.pi, len(params)),
    requires_grad=True, dtype=torch.float32
)

discriminator = Discriminator()
opt_g = optim.Adam([params_tensor], lr=0.01)
opt_d = optim.Adam(discriminator.parameters(), lr=0.005)
loss_fn = nn.BCELoss()
losses_d, losses_g = [], []

for epoch in range(1500):
    real = torch.tensor(generate_real_data(64)).float().view(-1, 1)
    real = (real - real.mean()) / real.std()
    fake_np = sample_from_generator(params_tensor.detach().numpy(), 64)
    fake = torch.tensor(fake_np).float()
    fake = (fake - fake.mean()) / fake.std()

    d_real = discriminator(real)
    d_fake = discriminator(fake)
    real_labels = torch.full_like(d_real, 0.9)
    fake_labels = torch.full_like(d_fake, 0.1)
    d_loss = loss_fn(d_real, real_labels) + loss_fn(d_fake, fake_labels)
    opt_d.zero_grad()
    d_loss.backward()
    opt_d.step()

    fake_np = sample_from_generator(params_tensor.detach().numpy(), 64)
    fake = torch.tensor(fake_np).float()
    fake = (fake - fake.mean()) / fake.std()
    d_fake = discriminator(fake)
    g_loss = loss_fn(d_fake, torch.ones_like(d_fake))
    opt_g.zero_grad()
    g_loss.backward()
    opt_g.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: D Loss = {d_loss.item():.4f}, G Loss = {g_loss.item():.4f}")
        losses_d.append(d_loss.item())
        losses_g.append(g_loss.item())

# Loss plot with axis labels and title
plt.plot(losses_d, label="Discriminator")
plt.plot(losses_g, label="Generator")
plt.title("Figure 1. Training Loss Curves for the QGAN Model")
plt.xlabel("Training Checkpoints (x100 Epochs)")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
plt.clf()

# Final histogram plot with axis labels
generated = sample_from_generator(params_tensor.detach().numpy(), 1000)
plt.hist(real_data, bins=50, alpha=0.5, label="Real", color="skyblue")
plt.hist(generated, bins=50, alpha=0.6, label="Generated", color="coral")
plt.title("Figure 2. Histogram: QGAN-Generated vs Real Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("final_output.png")

# KL divergence
real_hist, bins = np.histogram(real_data, bins=50, range=(-3, 3), density=True)
gen_hist, _ = np.histogram(generated, bins=50, range=(-3, 3), density=True)
kl = entropy(real_hist + 1e-8, gen_hist + 1e-8)
print(f"KL Divergence: {kl:.4f}")

# Save circuit as a PNG image
qc = create_generator(params)
circuit_drawer(qc, output="mpl", filename="quantum_generator_circuit.png")
