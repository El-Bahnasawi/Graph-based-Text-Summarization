# mm.py
# Weight tuning for SBERT-based summarization with upgraded, publication-ready plots.

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from Evaluator import SummarizationEvaluator  # If you use it elsewhere
from Summarizer import summarize_text         # If you use it elsewhere
from sent_embd import create_embedder

class WeightTuning:
    def __init__(self, dataset_path, sample_size=50, use_gpu=True):
        self.dataset_path = dataset_path
        self.sample_size = sample_size
        self.use_gpu = use_gpu
        self.results = []

    # ---------------------------
    # Public API
    # ---------------------------
    def run_weight_tuning(self):
        """Run weight tuning for SBERT with different parameters."""
        print("Starting weight tuning for SBERT...")

        # Parameter ranges
        sim_thresholds = np.arange(0.1, 1.0, 0.1)
        sent_name_weights = np.arange(0.0, 2.1, 0.2)
        name_name_weights = np.arange(0.0, 2.1, 0.2)

        # Load data
        print("Loading dataset...")
        df = pd.read_csv(self.dataset_path)
        df = df.head(self.sample_size)

        # Initialize SBERT embedder
        device = "cuda" if (self.use_gpu and self._cuda_available()) else "cpu"
        embedder = create_embedder("sbert", device=device)

        # Initialize ROUGE scorer
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        # Progress counters
        total_combinations = len(sim_thresholds) * len(sent_name_weights) * len(name_name_weights)
        current = 0

        # Evaluate each parameter combination
        for sim_threshold in sim_thresholds:
            for sent_name_weight in sent_name_weights:
                for name_name_weight in name_name_weights:
                    current += 1
                    print(
                        f"Evaluating {current}/{total_combinations}: "
                        f"sim_threshold={sim_threshold:.1f}, "
                        f"sent_name_weight={sent_name_weight:.1f}, "
                        f"name_name_weight={name_name_weight:.1f}"
                    )

                    config_results = {
                        'rouge1': [],
                        'rouge2': [],
                        'rougeL': [],
                        'rouge_l2': [],
                        'time': []
                    }

                    for i, row in df.iterrows():
                        article = row['article']
                        reference = row['highlights']

                        try:
                            start_time = time.time()

                            # Custom summarization with weight parameters
                            top_indices, summary_sentences = self.custom_summarize(
                                article,
                                embedder=embedder,
                                sim_threshold=sim_threshold,
                                sent_name_weight=sent_name_weight,
                                name_name_weight=name_name_weight
                            )

                            generation_time = time.time() - start_time
                            generated_summary = ' '.join(summary_sentences)

                            # Calculate ROUGE scores
                            scores = scorer.score(reference, generated_summary)
                            rouge_l2 = np.sqrt(
                                scores['rouge1'].fmeasure**2 +
                                scores['rouge2'].fmeasure**2 +
                                scores['rougeL'].fmeasure**2
                            )

                            config_results['rouge1'].append(scores['rouge1'].fmeasure)
                            config_results['rouge2'].append(scores['rouge2'].fmeasure)
                            config_results['rougeL'].append(scores['rougeL'].fmeasure)
                            config_results['rouge_l2'].append(rouge_l2)
                            config_results['time'].append(generation_time)

                        except Exception as e:
                            print(f"Error processing article {i}: {str(e)}")
                            continue

                    # Calculate average metrics
                    if len(config_results['rouge1']) == 0:
                        # No valid runs for this config
                        continue

                    avg_results = {
                        'sim_threshold': sim_threshold,
                        'sent_name_weight': sent_name_weight,
                        'name_name_weight': name_name_weight,
                        'avg_rouge1': float(np.mean(config_results['rouge1'])),
                        'avg_rouge2': float(np.mean(config_results['rouge2'])),
                        'avg_rougeL': float(np.mean(config_results['rougeL'])),
                        'avg_rouge_l2': float(np.mean(config_results['rouge_l2'])),
                        'avg_time': float(np.mean(config_results['time'])),
                        'n_articles': int(len(config_results['rouge1']))
                    }

                    self.results.append(avg_results)

        return self.results

    def custom_summarize(self, text, embedder, sim_threshold=0.5,
                         sent_name_weight=1.0, name_name_weight=1.0):
        """Custom summarization function with adjustable weights."""
        from Summarizer import split_into_sentences, _extract_names_per_sentence
        from Summarizer import build_graph_from_similarity  # not used, but kept for parity
        from reactive_graph import ReactiveGraph
        from Rank import PageRank

        # Split text into sentences
        sents = split_into_sentences(text)
        if not sents:
            return [], []

        # Generate embeddings
        X = embedder.encode(sents)

        # Calculate cosine similarity matrix
        def cosine_similarity_matrix(X):
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            S = np.matmul(Xn, Xn.T)
            np.fill_diagonal(S, 0.0)
            return S

        S = cosine_similarity_matrix(X.astype(np.float64))

        # Build graph
        g = ReactiveGraph(directed=False)

        # Add sentence nodes
        n = len(sents)
        for i in range(n):
            g.add_node(i, text=sents[i], kind="sent", value=1.0 / max(n, 1))

        # Add sentence-sentence edges if above threshold
        for i in range(n):
            for j in range(i + 1, n):
                w = float(S[i, j])
                if w >= sim_threshold:
                    g.add_edge(i, j, value=round(w, 3))

        # Extract names and add name-related edges with adjustable weights
        names_per_sent = _extract_names_per_sentence(sents)

        def custom_add_sentence_name_edges(g, sents, names_per_sent,
                                           sent_name_weight=1.0, name_name_weight=1.0):
            """Add sentence-name and name-name edges with given weights."""
            def _name_node_id(name):
                return f"NAME: {name}"

            # Sentence <-> Name edges
            for i, names in enumerate(names_per_sent):
                for nm in names:
                    nm = nm.strip()
                    if not nm:
                        continue
                    nid = _name_node_id(nm)
                    if nid not in g.nx:
                        g.add_node(nid, label=nm, kind="name", value=0.0)

                    # Merge or create edge with weight increment
                    if g.nx.has_edge(i, nid):
                        w = float(g.nx.edges[i, nid].get("value", 0.0)) + float(sent_name_weight)
                        g.set_edge_attrs(i, nid, value=round(w, 3))
                    elif g.nx.has_edge(nid, i):
                        w = float(g.nx.edges[nid, i].get("value", 0.0)) + float(sent_name_weight)
                        g.set_edge_attrs(nid, i, value=round(w, 3))
                    else:
                        g.add_edge(i, nid, value=float(sent_name_weight))

            # Name <-> Name edges aggregated per sentence
            from collections import Counter
            pair_counts = Counter()
            for names in names_per_sent:
                uniq = sorted(set(n.strip() for n in names if n.strip()))
                for a_idx in range(len(uniq)):
                    for b_idx in range(a_idx + 1, len(uniq)):
                        a, b = uniq[a_idx], uniq[b_idx]
                        pair_counts[(a, b)] += 1

            for (a, b), c in pair_counts.items():
                na, nb = _name_node_id(a), _name_node_id(b)
                if na not in g.nx:
                    g.add_node(na, label=a, kind="name", value=0.0)
                if nb not in g.nx:
                    g.add_node(nb, label=b, kind="name", value=0.0)

                weight_value = float(c) * float(name_name_weight)
                if g.nx.has_edge(na, nb):
                    w = float(g.nx.edges[na, nb].get("value", 0.0)) + weight_value
                    g.set_edge_attrs(na, nb, value=round(w, 3))
                elif g.nx.has_edge(nb, na):
                    w = float(g.nx.edges[nb, na].get("value", 0.0)) + weight_value
                    g.set_edge_attrs(nb, na, value=round(w, 3))
                else:
                    g.add_edge(na, nb, value=weight_value)

        custom_add_sentence_name_edges(
            g, sents, names_per_sent,
            sent_name_weight=sent_name_weight,
            name_name_weight=name_name_weight
        )

        # Run PageRank
        pr = PageRank()
        res = pr.rank(
            g,
            damping=0.85,
            max_iter=50,
            tol=1e-6,
            sort_desc=True,
            live_update=False,
            update_attr="value",
            sleep_between_iters=0
        )

        # Select top sentences
        sent_scores = {i: s for i, s in res.scores.items() if isinstance(i, int)}
        top_sorted = sorted(sent_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        top_ids = sorted([i for i, _ in top_sorted])
        top_sentences = [sents[i] for i in top_ids]

        return top_ids, top_sentences

    def plot_results(self):
        """Summarize and visualize weight tuning results with improved plots."""
        if not getattr(self, "results", None):
            print("No results to plot!")
            return

        self._set_plot_style()
        df = pd.DataFrame(self.results)
        df = df.dropna(subset=["avg_rouge1", "avg_rouge2", "avg_rougeL", "avg_rouge_l2"]).copy()
        if df.empty:
            print("Results are empty after cleaning.")
            return

        # Identify best configuration by ROUGE L2 norm
        best_idx = df["avg_rouge_l2"].idxmax()
        best_config = df.loc[best_idx]
        print("\nBest configuration:")
        print(f"  sim_threshold: {best_config['sim_threshold']:.1f}")
        print(f"  sent_name_weight: {best_config['sent_name_weight']:.1f}")
        print(f"  name_name_weight: {best_config['name_name_weight']:.1f}")
        print(f"  ROUGE L2-Norm: {best_config['avg_rouge_l2']:.4f}")
        if "avg_time" in df.columns and pd.notna(best_config.get("avg_time", np.nan)):
            print(f"  Avg time/article: {best_config['avg_time']:.3f}s")

        # Heatmaps across parameter interactions
        self._plot_heatmaps(df)
        # Radar chart comparing best slices and overall best
        self._plot_radar_chart(df, best_config)

        # Save CSV for downstream analysis
        out = df.sort_values("avg_rouge_l2", ascending=False)
        out.to_csv("weight_tuning_results.csv", index=False)
        print("Results saved to weight_tuning_results.csv")

    # ---------------------------
    # Plotting helpers
    # ---------------------------
    def _set_plot_style(self):
        """Consistent, legible plotting defaults."""
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except Exception:
            plt.style.use("ggplot")
        plt.rcParams.update({
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "legend.fontsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        })

    def _plot_heatmaps(self, df: pd.DataFrame):
        """Create normalized heatmaps for parameter interactions with consistent scales."""
        vmin = float(df["avg_rouge_l2"].min())
        vmax = float(df["avg_rouge_l2"].max())

        def _pivot(y, x):
            return df.pivot_table(values="avg_rouge_l2", index=y, columns=x, aggfunc="mean")

        pivots = [
            ("sim_threshold", "sent_name_weight",
             "ROUGE L2: sim_threshold vs sent_name_weight", "sent_name_weight", "sim_threshold"),
            ("sim_threshold", "name_name_weight",
             "ROUGE L2: sim_threshold vs name_name_weight", "name_name_weight", "sim_threshold"),
            ("sent_name_weight", "name_name_weight",
             "ROUGE L2: sent_name_weight vs name_name_weight", "name_name_weight", "sent_name_weight"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
        cmap = "cividis"
        for ax, (iy, ix, title, xlabel, ylabel) in zip(axes, pivots):
            pivot = _pivot(iy, ix)
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={"label": "Avg ROUGE L2"},
                annot_kws={"size": 10},
                ax=ax,
            )
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=45)
            ax.tick_params(axis="y", rotation=0)
            ax.xaxis.set_major_locator(MaxNLocator(nbins="auto", integer=False, prune=None))
            ax.yaxis.set_major_locator(MaxNLocator(nbins="auto", integer=False, prune=None))

        fig.suptitle("Parameter interaction heatmaps (normalized scale)", y=1.02)
        plt.savefig("weight_tuning_heatmaps.png", bbox_inches="tight")
        plt.close(fig)

    def _plot_radar_chart(self, df: pd.DataFrame, best_config: pd.Series):
        """Polar radar comparing best overall vs per-parameter best averages.
        Includes a speed score if timing is available (normalized so higher is faster).
        """
        # Best-by-group frames
        best_sim = df.loc[df.groupby("sim_threshold")["avg_rouge_l2"].idxmax()]
        best_sent_name = df.loc[df.groupby("sent_name_weight")["avg_rouge_l2"].idxmax()]
        best_name_name = df.loc[df.groupby("name_name_weight")["avg_rouge_l2"].idxmax()]

        categories = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "ROUGE L2"]

        def _mean_safe(obj, col):
            # Works for both Series (single row) and DataFrame slices
            if hasattr(obj, "mean"):
                return float(obj[col].mean())
            return float(obj[col])

        best_overall_vals = [
            float(best_config["avg_rouge1"]),
            float(best_config["avg_rouge2"]),
            float(best_config["avg_rougeL"]),
            float(best_config["avg_rouge_l2"]),
        ]
        best_sim_vals = [
            _mean_safe(best_sim, "avg_rouge1"),
            _mean_safe(best_sim, "avg_rouge2"),
            _mean_safe(best_sim, "avg_rougeL"),
            _mean_safe(best_sim, "avg_rouge_l2"),
        ]
        best_sent_name_vals = [
            _mean_safe(best_sent_name, "avg_rouge1"),
            _mean_safe(best_sent_name, "avg_rouge2"),
            _mean_safe(best_sent_name, "avg_rougeL"),
            _mean_safe(best_sent_name, "avg_rouge_l2"),
        ]
        best_name_name_vals = [
            _mean_safe(best_name_name, "avg_rouge1"),
            _mean_safe(best_name_name, "avg_rouge2"),
            _mean_safe(best_name_name, "avg_rougeL"),
            _mean_safe(best_name_name, "avg_rouge_l2"),
        ]

        # Optional speed score
        speed_available = "avg_time" in df.columns and df["avg_time"].notna().any()
        if speed_available:
            t = df["avg_time"].astype(float)
            tmin, tmax = float(t.min()), float(t.max())
            if tmax > tmin:
                def speed_score(x):
                    return 1.0 - (float(x) - tmin) / (tmax - tmin)
                categories.append("Speed")
                best_overall_vals.append(speed_score(best_config.get("avg_time", tmax)))
                best_sim_vals.append(speed_score(_mean_safe(best_sim, "avg_time")))
                best_sent_name_vals.append(speed_score(_mean_safe(best_sent_name, "avg_time")))
                best_name_name_vals.append(speed_score(_mean_safe(best_name_name, "avg_time")))

        # Radar plot
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, polar=True)
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        def plot_radar(values, label):
            vals = values + values[:1]
            ax.plot(angles, vals, linewidth=2.5, label=label)
            ax.fill(angles, vals, alpha=0.25)

        plot_radar(best_overall_vals, "Best overall")
        plot_radar(best_sim_vals, "Best by sim_threshold")
        plot_radar(best_sent_name_vals, "Best by sent_name_weight")
        plot_radar(best_name_name_vals, "Best by name_name_weight")

        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_title("Best configurations comparison")
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05))
        ax.set_rlabel_position(30)
        ax.grid(True)
        ax.set_ylim(0.0, 1.0)  # ROUGE metrics and normalized speed all in 0..1

        plt.savefig("weight_tuning_radar.png", bbox_inches="tight")
        plt.close(fig)

    # ---------------------------
    # Utilities
    # ---------------------------
    @staticmethod
    def _cuda_available():
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False


if __name__ == "__main__":
    # Run weight tuning end to end, then plot results
    tuning = WeightTuning(
        dataset_path="Datasets/cnn_dailymail/validation.csv",
        sample_size=50,  # keep small for faster tuning; increase when ready
        use_gpu=True
    )
    results = tuning.run_weight_tuning()
    tuning.plot_results()
