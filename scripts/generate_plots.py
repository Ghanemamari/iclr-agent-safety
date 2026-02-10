"""
Generate publication-quality plots for the prompt injection detection paper.
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
import os

# Use a clean style for publications
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.dpi'] = 150

# Colors - colorblind friendly palette
COLORS = {
    'probe': '#2ecc71',      # Green
    'semantic': '#3498db',   # Blue  
    'statistical': '#e74c3c', # Red
    'tfidf': '#9b59b6',      # Purple
}

def load_results():
    """Load all benchmark results."""
    results = {}
    
    # Stealthy dataset results (the key benchmark)
    stealthy_dir = "data/processed_hard"
    
    # Try loading probe metrics
    probe_files = [f for f in os.listdir(stealthy_dir) if f.endswith("_metrics.json") and "TinyLlama" in f]
    if probe_files:
        with open(os.path.join(stealthy_dir, probe_files[0])) as f:
            results['probe_stealthy'] = json.load(f)
    
    # Statistical baseline
    stat_path = os.path.join(stealthy_dir, "statistical_metrics.json")
    if os.path.exists(stat_path):
        with open(stat_path) as f:
            results['statistical'] = json.load(f)
            
    # Semantic baseline
    sem_path = os.path.join(stealthy_dir, "semantic_metrics.json")
    if os.path.exists(sem_path):
        with open(sem_path) as f:
            results['semantic'] = json.load(f)
    
    return results

def plot_method_comparison():
    """Bar chart comparing AUC across methods on Stealthy dataset."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Data from our experiments (Stealthy dataset - the hardest test)
    methods = ['Linear Probe\n(Ours)', 'TF-IDF +\nLogistic Reg.', 'Sentence\nEmbeddings', 'Perplexity\nBaseline']
    aucs = [0.99, 0.45, 0.25, 0.28]  # From stealthy dataset results
    colors = [COLORS['probe'], COLORS['tfidf'], COLORS['semantic'], COLORS['statistical']]
    
    bars = ax.bar(methods, aucs, color=colors, edgecolor='black', linewidth=0.8, width=0.6)
    
    # Add value labels on bars
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax.annotate(f'{auc:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('AUC-ROC')
    ax.set_ylim(0, 1.15)
    ax.set_title('Method Comparison on Stealthy Dataset', fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Random baseline')
    ax.legend(loc='upper right')
    
    # Add subtle grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('data/plots/method_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('data/plots/method_comparison.png', bbox_inches='tight', dpi=300)
    print("Saved: method_comparison.pdf/png")
    return fig

def plot_layer_analysis():
    """Line plot showing AUC across different transformer layers."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Data from layer-wise analysis (TinyLlama on Complex dataset)
    layers = [0, 5, 11, 16, 21]
    aucs_tinyllama = [0.70, 0.98, 0.99, 1.00, 1.00]  # From comprehensive report
    aucs_qwen = [0.99, 0.99, 0.99, 0.99, 0.99]  # Qwen shows consistent signal
    
    ax.plot(layers, aucs_tinyllama, 'o-', color=COLORS['probe'], linewidth=2, 
            markersize=8, label='TinyLlama-1.1B', markeredgecolor='black', markeredgewidth=0.5)
    ax.plot(layers, aucs_qwen, 's--', color=COLORS['semantic'], linewidth=2,
            markersize=8, label='Qwen2.5-0.5B', markeredgecolor='black', markeredgewidth=0.5)
    
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Layer-wise Probe Performance', fontweight='bold')
    ax.set_ylim(0.6, 1.05)
    ax.set_xticks(layers)
    ax.legend(loc='lower right')
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Shaded region for "good" performance
    ax.axhspan(0.95, 1.05, alpha=0.1, color='green')
    ax.text(1, 1.02, 'Excellent', fontsize=9, color='green', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('data/plots/layer_analysis.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('data/plots/layer_analysis.png', bbox_inches='tight', dpi=300)
    print("Saved: layer_analysis.pdf/png")
    return fig

def plot_model_comparison():
    """Grouped bar chart comparing models across datasets."""
    fig, ax = plt.subplots(figsize=(7, 4))
    
    models = ['TinyLlama\n(1.1B)', 'Qwen\n(0.5B)']
    x = np.arange(len(models))
    width = 0.35
    
    # AUC scores from our experiments
    stealthy_scores = [1.00, 1.00]
    complex_scores = [1.00, 0.997]
    
    bars1 = ax.bar(x - width/2, stealthy_scores, width, label='Stealthy Dataset', 
                   color=COLORS['probe'], edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, complex_scores, width, label='Complex Dataset',
                   color=COLORS['semantic'], edgecolor='black', linewidth=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Model Generalization Across Datasets', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.9, 1.05)
    ax.legend(loc='lower right')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('data/plots/model_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('data/plots/model_comparison.png', bbox_inches='tight', dpi=300)
    print("Saved: model_comparison.pdf/png")
    return fig

def plot_statistical_significance():
    """Horizontal bar chart showing effect sizes and confidence."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    
    metrics = ["Cohen's d", "AUC (95% CI)"]
    values = [10.57, 1.00]
    errors = [[0], [0]]  # CI is [1.00, 1.00]
    
    colors = [COLORS['probe'], COLORS['semantic']]
    
    y_pos = np.arange(len(metrics))
    bars = ax.barh(y_pos, values, color=colors, edgecolor='black', height=0.5)
    
    # Add annotations
    ax.text(10.57 + 0.3, 0, 'd = 10.57\n(Massive effect)', va='center', fontsize=10)
    ax.text(1.0 + 0.1, 1, 'AUC = 1.00\n[1.00, 1.00]', va='center', fontsize=10)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.set_xlabel('Value')
    ax.set_title('Statistical Significance (Stealthy Dataset, N=400)', fontweight='bold')
    ax.set_xlim(0, 14)
    
    # Add reference lines
    ax.axvline(x=0.8, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(0.85, -0.4, "Large effect\nthreshold (d=0.8)", fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.savefig('data/plots/statistical_significance.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('data/plots/statistical_significance.png', bbox_inches='tight', dpi=300)
    print("Saved: statistical_significance.pdf/png")
    return fig

if __name__ == "__main__":
    # Create output directory
    os.makedirs("data/plots", exist_ok=True)
    
    print("Generating publication-quality plots...")
    print("=" * 50)
    
    plot_method_comparison()
    plot_layer_analysis()
    plot_model_comparison()
    plot_statistical_significance()
    
    print("=" * 50)
    print("All plots saved to data/plots/")
    print("Use the .pdf files for LaTeX and .png for previews.")
