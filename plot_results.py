import matplotlib.pyplot as plt
import numpy as np

# Impostazione dello stile visivo per pubblicazioni scientifiche
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 16
})

# =========================================================
# FIGURA 1: Confronto Tempi di Esecuzione (CPU vs GPU)
# =========================================================
labels = ['CPU (MPI+OpenMP\n32 Cores)', 'GPU (CUDA\nQuadro)']
execution_times = [17.458532, 0.065048]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(labels, execution_times, color=['#2b5c8f', '#2ca02c'], width=0.5)

# Scala logaritmica per apprezzare la differenza di ordine di grandezza
ax.set_yscale('log')
ax.set_ylabel('Tempo di Esecuzione in Secondi (Scala Log)')
ax.set_title('Confronto Prestazioni: CPU vs GPU (test_02)', pad=15)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Etichette sui punti delle barre
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval * 1.3, f'{yval:.3f} s', 
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('cpu_vs_gpu.png', dpi=300)
plt.show()

# =========================================================
# FIGURA 2: Strong Scaling & Speedup OpenMP (Singolo Nodo)
# =========================================================
threads = np.array([1, 2, 4, 8, 16, 32])
# Esempio di tempi tipici al variare dei thread (sostituisci con i tuoi dati reali)
times_omp = np.array([210.0, 108.5, 55.2, 28.1, 19.5, 17.45]) 

# Speedup effettivo vs Speedup Ideale
speedup_real = times_omp[0] / times_omp
speedup_ideal = threads

fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot Speedup
color1 = '#1f77b4'
ax1.set_xlabel('Numero di Thread OpenMP (Core CPU)')
ax1.set_ylabel('Speedup ($T_1 / T_N$)', color=color1)
line1 = ax1.plot(threads, speedup_real, 'o-', color=color1, linewidth=2, label='Speedup Misurato')
line2 = ax1.plot(threads, speedup_ideal, '--', color='gray', alpha=0.7, label='Speedup Ideale (Lineare)')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xticks(threads)
ax1.grid(True, linestyle=':', alpha=0.6)

# Secondo asse per l'Efficienza Parallela
ax2 = ax1.twinx()
color2 = '#d62728'
efficiency = (speedup_real / threads) * 100
ax2.set_ylabel('Efficienza Parallela (%)', color=color2)
line3 = ax2.plot(threads, efficiency, 's--', color=color2, linewidth=1.5, label='Efficienza (%)')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, 110)

# Unione delle leggende
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

plt.title('Strong Scaling OpenMP & Efficienza Parallela', pad=15)
plt.tight_layout()
plt.savefig('strong_scaling.png', dpi=300)
plt.show()

print("Grafici generati con successo: 'cpu_vs_gpu.png' e 'strong_scaling.png'")