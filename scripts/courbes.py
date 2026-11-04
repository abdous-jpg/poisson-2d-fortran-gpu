"""Trace les courbes a partir des CSV de results/ (lancer depuis la racine du depot)."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Convergence
c = pd.read_csv("results/convergence.csv")
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.loglog(c["h"], c["erreur_l2"], "o-", label="Jacobi (Fortran)")
ref = c["erreur_l2"].iloc[0] * (c["h"] / c["h"].iloc[0]) ** 2
ax.loglog(c["h"], ref, "--", color="gray", label="pente 2")
ax.set_xlabel("pas h")
ax.set_ylabel("erreur L2")
ax.set_title("Convergence sur solution exacte : ordre 2")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("results/convergence.png", dpi=150)

# 2. Performance : debit (millions de points mis a jour par seconde)
p = pd.read_csv("results/performance.csv")
fig, ax = plt.subplots(figsize=(6.5, 4))
for (prec, t), g in p.groupby(["precision", "threads"]):
    ax.plot(g["n"], g["mpoints_par_s"], "o-", label=f"{prec} precision, {t} thread" + ("s" if t > 1 else ""))
ax.set_xscale("log", base=2)
ax.set_xticks(sorted(p["n"].unique()), [str(v) for v in sorted(p["n"].unique())])
ax.set_xlabel("taille de grille n (n x n points)")
ax.set_ylabel("Mpoints / s")
ax.set_title("Debit du solveur de Jacobi sur CPU (OpenMP)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("results/performance.png", dpi=150)
print("Courbes ecrites dans results/")
