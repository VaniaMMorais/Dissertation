import json
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 1. Carregar os novos resultados (com o nome correto do ficheiro)
with open('ragas_official_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

detailed_results = data.get('detailed_results', [])

# As 4 métricas oficiais
metric_names = ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']

# Função auxiliar para ignorar NaNs de forma segura
def is_valid(x):
    return isinstance(x, (int, float)) and not math.isnan(x)

# ==========================================
# GRÁFICO 1: MÉTRICAS GERAIS (OVERALL)
# ==========================================
means = []
stds = []

for metric in metric_names:
    vals = [res['metrics'].get(metric) for res in detailed_results]
    clean_vals = [v for v in vals if is_valid(v)]
    
    if clean_vals:
        means.append(np.mean(clean_vals))
        stds.append(np.std(clean_vals))
    else:
        means.append(0)
        stds.append(0)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metric_names))

# Desenhar as barras
bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color='steelblue', edgecolor='black')

# Adicionar linhas de threshold
ax.axhline(y=0.80, color='green', linestyle='--', label='Bom (0.80)')
ax.axhline(y=0.70, color='orange', linestyle='--', label='Aceitável (0.70)')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('RAGAS Evaluation Metrics (Overall)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names], fontsize=11)
ax.set_ylim(0, 1.1) # Ligeiramente acima de 1.0 para caberem as barras de erro
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Adicionar o número exato no topo de cada barra para clareza
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('ragas_overall_metrics.png', dpi=300)
print("✅ Gráfico 1 guardado: ragas_overall_metrics.png")

# ==========================================
# GRÁFICO 2: MÉTRICAS POR CATEGORIA
# ==========================================
# Agrupar os dados por categoria dinamicamente
categories_data = defaultdict(lambda: defaultdict(list))

for res in detailed_results:
    cat = res.get('category', 'Unknown')
    for metric in metric_names:
        val = res['metrics'].get(metric)
        if is_valid(val):
            categories_data[cat][metric].append(val)

cat_names = list(categories_data.keys())

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(cat_names))
width = 0.2  # Ajustado para caberem 4 barras lado a lado

# Cores distintas para cada métrica
colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b2']

for i, metric in enumerate(metric_names):
    metric_means = []
    for cat in cat_names:
        vals = categories_data[cat][metric]
        metric_means.append(np.mean(vals) if vals else 0)
    
    # Calcular o offset para as barras não ficarem sobrepostas
    offset = (i - len(metric_names)/2 + 0.5) * width
    ax.bar(x + offset, metric_means, width, label=metric.replace('_', ' ').title(), color=colors[i], alpha=0.85, edgecolor='black')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('RAGAS Metrics by Query Category', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([c.replace('_', ' ').title() for c in cat_names], fontsize=11)
ax.set_ylim(0, 1.1)

# Colocar a legenda de lado para não tapar os dados
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('ragas_by_category.png', dpi=300)
print("✅ Gráfico 2 guardado: ragas_by_category.png")

plt.show()