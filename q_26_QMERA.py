"""
QMERA - Quantum Multi-scale Entanglement Renormalization Ansatz
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize as scipy_minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals
 
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
MAXITER = 250


def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def mera_circuit(theta):
    qc = QuantumCircuit(NUM_QUBITS)

    for i in range(NUM_QUBITS):
        qc.h(i)

    idx = 0

    for i in range(0, NUM_QUBITS - 1, 2):
        qc.ry(theta[idx], i)
        qc.ry(theta[idx + 1], i + 1)
        qc.cx(i, i + 1)
        qc.rz(theta[idx + 2], i + 1)
        idx += 3

    for i in range(1, NUM_QUBITS - 1, 2):
        qc.ry(theta[idx], i)
        qc.ry(theta[idx + 1], i + 1)
        qc.cx(i, i + 1)
        qc.rz(theta[idx + 2], i + 1)
        idx += 3

    for i in range(NUM_QUBITS):
        qc.ry(theta[idx], i)
        idx += 1
    for i in range(NUM_QUBITS - 1):
        qc.cx(i, i + 1)
    qc.cx(NUM_QUBITS - 1, 0)
    for i in range(NUM_QUBITS):
        qc.rz(theta[idx], i)
        idx += 1

    for i in range(0, NUM_QUBITS - 1, 2):
        qc.ry(theta[idx], i)
        qc.ry(theta[idx + 1], i + 1)
        qc.cx(i, i + 1)
        qc.rz(theta[idx + 2], i + 1)
        idx += 3

    for i in range(NUM_QUBITS):
        qc.ry(theta[idx], i)
        idx += 1

    return qc, idx


def num_mera_params():
    _, n = mera_circuit(np.zeros(100))
    return n


def train_mera(target):
    n_p = num_mera_params()
    theta0 = np.random.uniform(0, 2 * np.pi, n_p)

    def cost(theta):
        qc, _ = mera_circuit(theta)
        sv = Statevector.from_instruction(qc)
        born = sv.probabilities()
        kl = 0.0
        for i, pt in enumerate(target):
            if pt > 0:
                pb = max(born[i], 1e-10)
                kl += pt * np.log(pt / pb)
        return float(kl)

    res = scipy_minimize(cost, theta0, method='COBYLA',
                         options={'maxiter': MAXITER, 'rhobeg': 0.5})
    return res.x, res.fun


def generate_dist(theta):
    qc, _ = mera_circuit(theta)
    sv = Statevector.from_instruction(qc)
    return sv.probabilities()


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, score in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    n_p = num_mera_params()
    print(f"\n--- QMERA ({NUM_QUBITS}q, MERA ansatz, {n_p} params, "
          f"COBYLA {MAXITER} iter) ---")

    dists = []
    for pos in range(7):
        print(f"  Poz {pos+1}...", end=" ", flush=True)
        target = build_empirical(draws, pos)
        theta, loss = train_mera(target)
        born = generate_dist(theta)
        dists.append(born)

        top_idx = np.argsort(born)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{born[i]:.3f}" for i in top_idx)
        print(f"KL={loss:.4f}  top: {info}")

    combo = greedy_combo(dists)

    print(f"\n{'='*50}")
    print(f"Predikcija (QMERA, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()



"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- QMERA (5q, MERA ansatz, 33 params, COBYLA 250 iter) ---
  Poz 1... KL=0.0162  top: 1:0.176 | 2:0.168 | 3:0.132
  Poz 2... KL=0.0497  top: 8:0.086 | 7:0.078 | 6:0.075
  Poz 3... KL=0.0610  top: 16:0.083 | 15:0.079 | 13:0.067
  Poz 4... KL=0.0235  top: 23:0.069 | 18:0.066 | 21:0.058
  Poz 5... KL=0.0144  top: 26:0.064 | 25:0.062 | 21:0.060
  Poz 6... KL=0.0298  top: 31:0.078 | 34:0.076 | 33:0.074
  Poz 7... KL=0.1466  top: 38:0.163 | 37:0.099 | 7:0.093

==================================================
Predikcija (QMERA, deterministicki, seed=39):
[1, 8, x, y, z, 31, 38]
==================================================
"""



"""
QMERA - Quantum Multi-scale Entanglement Renormalization Ansatz

MERA je kvantni analog KSR (Kadanoff-Baym-Renormalization Ansatz) za klasične sisteme
MERA koristi kvantnu evoluciju za aproksimaciju Born distribucije
MERA se sastoji od 5 qubita i 3 sloja Ry+CX+Rz rotacija

MERA topologija: hijerarhijski entanglement inspirisan renormalizacionom grupom iz fizike
Sloj 1: paran-neparan disentangler blokovi (Ry+CX+Rz na parovima 0-1, 2-3)
Sloj 2: neparan-paran blokovi (parovi 1-2, 3-4) - "isotropic" sloj
Sloj 3: globalni ciklicni entanglement (Ry+CX ring+Rz)
Sloj 4: ponovo paran-neparan disentangler
Sloj 5: finalne Ry rotacije
Hvata korelacije na razlicitim skalama - lokalne i globalne istovremeno
KL divergencija, COBYLA 250 iteracija
Inspirisano tensor network MERA iz kvantne fizike kondenzovane materije
Deterministicki, Statevector. 
"""
