import json
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 1. Carregar os resultados
with open('ragas_official_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

detailed_results = data.get('detailed_results', [])

def is_valid(x):
    """Verifica se o número é válido (ignora NaNs)"""
    return isinstance(x, (int, float)) and not math.isnan(x)

# ==========================================
# CALCULAR O F1-SCORE PARA CADA PERGUNTA
# ==========================================
for res in detailed_results:
    p = res['metrics'].get('context_precision', 0)
    r = res['metrics'].get('context_recall', 0)
    
    # Se ambos os valores forem válidos e a soma for maior que 0 (evita dividir por zero)
    if is_valid(p) and is_valid(r) and (p + r) > 0:
        f1 = 2 * (p * r) / (p + r)
    else:
        f1 = 0.0
    
    res['metrics']['context_f1'] = f1

# As 5 métricas que vamos desenhar
metric_names = ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision', 'context_f1']

# ==========================================
# GRÁFICO 1: MÉTRICAS GERAIS (OVERALL)
# ==========================================
means = []
stds = []

for metric in metric_names:
    vals = [res['metrics'].get(metric) for res in detailed_results if is_valid(res['metrics'].get(metric))]
    if vals:
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    else:
        means.append(0)
        stds.append(0)

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(metric_names))

# Desenhar as barras
bars = ax.bar(x, means, alpha=0.8, color='steelblue', edgecolor='black')

# Adicionar linhas de threshold
ax.axhline(y=0.80, color='green', linestyle='--', label='Bom (0.80)')
ax.axhline(y=0.70, color='orange', linestyle='--', label='Aceitável (0.70)')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('RAGAS Evaluation Metrics (Overall)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
labels = ['Faithfulness', 'Answer\nRelevancy', 'Context\nRecall', 'Context\nPrecision', 'Context\nF1-Score']
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1.15) # Ajustado para margens de erro
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Valores no topo
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
categories_data = defaultdict(lambda: defaultdict(list))

for res in detailed_results:
    cat = res.get('category', 'Unknown')
    for metric in metric_names:
        val = res['metrics'].get(metric)
        if is_valid(val):
            categories_data[cat][metric].append(val)

cat_names = list(categories_data.keys())

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(cat_names))
width = 0.16  # Um pouco mais fino para caberem 5 barras

# 5 cores distintas (adicionei um dourado para o F1)
colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b2', '#ccb974']
legend_labels = ['Faithfulness', 'Answer Relevancy', 'Context Recall', 'Context Precision', 'Context F1-Score']

for i, metric in enumerate(metric_names):
    metric_means = []
    for cat in cat_names:
        vals = categories_data[cat][metric]
        metric_means.append(np.mean(vals) if vals else 0)
    
    offset = (i - len(metric_names)/2 + 0.5) * width
    ax.bar(x + offset, metric_means, width, label=legend_labels[i], color=colors[i], alpha=0.85, edgecolor='black')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('RAGAS Metrics by Query Category', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([c.replace('_', ' ').title() for c in cat_names], fontsize=11)
ax.set_ylim(0, 1.15)

ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('ragas_by_category.png', dpi=300)
print("✅ Gráfico 2 guardado: ragas_by_category.png")