import matplotlib.pyplot as plt
import pandas as pd
import os

csv_dir = "results_csv"
marker_size = 8

# Ordre désiré des courbes (doit correspondre aux suffixes des noms de fichiers)
scenario_order = [
    "clusters_6_7_8_blocked",
    "clusters_2_3_4_blocked",
    "cluster_1_blocked",
    "cluster_5_blocked",
    "cluster_9_blocked",
    "cluster_24_blocked",
    "no_blockage"
]

cluster_labels = {
    "clusters_6_7_8_blocked": "{6,7,8}",
    "clusters_2_3_4_blocked": "{2,3,4}",
    "cluster_1_blocked": "{1}",
    "cluster_5_blocked": "{5}",
    "cluster_9_blocked": "{9}",
    "cluster_24_blocked": "{24}",
    "no_blockage": r"\mathbf{NoBlk}",
}

colors = [
    "#ff7f0e", "#17becf", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#1f77b4"
]

fig, ax = plt.subplots(figsize=(6, 5))
ax.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=14)
ax.set_ylabel("BER", fontsize=14)
ax.set_yscale("log")
ax.set_ylim([1e-3, 0.6])
ax.grid(which="both")

for i, scenario in enumerate(scenario_order):
    filename = f"BER_CDL_C_{scenario}.csv"
    filepath = os.path.join(csv_dir, filename)
    df = pd.read_csv(filepath)

    label_base = cluster_labels[scenario]
    label_theo = f"Theo, BlkClust=${label_base}$"
    label_mc = f"MC, BlkClust=${label_base}$"

    ax.semilogy(df["EbN0_dB"], df["BER_MC"], "*", markersize=marker_size, label=label_mc, color=colors[i])
    ax.semilogy(df["EbN0_dB"], df["BER_theoretical"], "-", linewidth=2, label=label_theo, color=colors[i])

ax.legend(loc="best", fontsize=10, frameon=True)
plt.tight_layout()
plt.title("BER Monte-Carlo & Théorique - Scénarios CDL-C")
plt.show()
