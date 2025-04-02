
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

# Fichiers à exécuter
files = ["MAB_e.py", "MAB_u.py"]
results = {}

for file in files:
    result = subprocess.run(["python3", file], capture_output=True, text=True)
    print(f"\nRésultats pour {file}:")
    print(result.stdout)

    # Extraction des valeurs finales
    lines = result.stdout.strip().split("\n")
    values_line = [l for l in lines if "Valeurs estimées" in l][0]
    counts_line = [l for l in lines if "Nombre de fois" in l][0]

    values = eval(values_line.split(":")[-1].strip())
    counts = eval(counts_line.split(":")[-1].strip())
    results[file] = {"values": values, "counts": counts}

# Comparaison graphique
labels = ["Direct", "Edge Cloud"]
x = range(len(labels))

plt.figure(figsize=(10, 4))

# Valeurs estimées
plt.subplot(1, 2, 1)
for file, color in zip(files, ['blue', 'green']):
    plt.bar(x, results[file]["values"], alpha=0.6, label=file, color=color)
plt.title("Valeurs estimées")
plt.xticks(x, labels)
plt.ylabel("Reward")
plt.legend()

# Nombre de fois sélectionné
plt.subplot(1, 2, 2)
for file, color in zip(files, ['blue', 'green']):
    plt.bar(x, results[file]["counts"], alpha=0.6, label=file, color=color)
plt.title("Nombre de sélections")
plt.xticks(x, labels)
plt.ylabel("Count")
plt.legend()

plt.tight_layout()
plt.show()
