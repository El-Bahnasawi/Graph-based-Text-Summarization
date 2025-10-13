import sys
import os
import json
import time
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rouge_score import rouge_scorer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from summarizer import GraphSummarizer, SummaryConfig
from sent_embd import create_embedder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for ablation study experiments"""
    name: str
    embedder_kind: str
    sim_threshold: float
    use_sent_sent: bool
    use_sent_name: bool
    use_name_name: bool
    description: str = ""


class GraphBuilder(ABC):
    """Abstract base class for graph building strategies"""
    
    @abstractmethod
    def build_config(self, config: AblationConfig) -> SummaryConfig:
        pass


class AblationGraphBuilder(GraphBuilder):
    """Concrete graph builder for ablation study configurations"""
    
    def build_config(self, config: AblationConfig) -> SummaryConfig:
        """Build SummaryConfig from AblationConfig"""
        return SummaryConfig(
            k=3,  # Number of sentences for summary
            sim_threshold=config.sim_threshold,
            embedder_kind=config.embedder_kind,
            use_sentence_edges=config.use_sent_sent,
            use_sent_name_edges=config.use_sent_name,
            use_name_name_edges=config.use_name_name,
            damping=0.85,
            max_iter=50
        )


class ResultExporter(ABC):
    """Abstract base class for result exporters"""
    
    @abstractmethod
    def export(self, results: List[Dict], output_dir: str) -> Dict[str, str]:
        pass


class CSVResultExporter(ResultExporter):
    """Exports results to CSV format"""
    
    def export(self, results: List[Dict], output_dir: str) -> Dict[str, str]:
        """Export results to CSV files"""
        if not results:
            return {}
        
        df = pd.DataFrame(results)
        exported_files = {}
        
        try:
            # Main results
            results_csv_path = os.path.join(output_dir, "ablation_results.csv")
            df.to_csv(results_csv_path, index=False)
            exported_files['results_csv'] = results_csv_path
            
            # Detailed results with metadata
            detailed_csv_path = os.path.join(output_dir, "ablation_results_detailed.csv")
            detailed_df = df.copy()
            detailed_df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            detailed_df.to_csv(detailed_csv_path, index=False)
            exported_files['detailed_csv'] = detailed_csv_path
            
            return exported_files
            
        except Exception as e:
            logger.error(f"Failed to export CSV results: {e}")
            return {}


class PlotGenerator(ABC):
    """Abstract base class for plot generators"""
    
    @abstractmethod
    def generate_plot(self, df: pd.DataFrame, output_path: str) -> str:
        pass


class RougePlotGenerator(PlotGenerator):
    """Generates ROUGE scores comparison plot"""
    
    def generate_plot(self, df: pd.DataFrame, output_path: str) -> str:
        """Generate ROUGE scores comparison plot"""
        fig, ax = plt.subplots(figsize=(14, 8))
        self._plot_rouge_comparison(ax, df)
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return output_path
    
    def _plot_rouge_comparison(self, ax, df: pd.DataFrame) -> None:
        """Plot ROUGE scores comparison across configurations"""
        x_pos = np.arange(len(df))
        bar_width = 0.2
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        metrics = ['avg_rouge1', 'avg_rouge2', 'avg_rougeL', 'avg_rouge_l2']
        labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE L2-Norm']
        
        for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
            ax.bar(x_pos + (i-1.5)*bar_width, df[metric], bar_width, 
                   label=label, alpha=0.8, color=color, yerr=df[f'std_{metric[4:]}'])

        ax.set_xlabel('Configuration', fontsize=14, fontweight='bold')
        ax.set_ylabel('ROUGE Score', fontsize=14, fontweight='bold')
        ax.set_title('Ablation Study: ROUGE Scores by Configuration', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([row['config_name'] for _, row in df.iterrows()], 
                          rotation=45, ha='right', fontsize=10)
        ax.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(df['avg_rouge_l2']) * 1.3)


class AblationStudy:
    """
    Comprehensive ablation study for graph-based text summarization.
    
    This class systematically evaluates the contribution of different
    graph components and embedding methods to the overall performance.
    """
    
    def __init__(self, dataset_path: str, sample_size: int = 100, use_gpu: bool = True, 
                 output_dir: str = "results/ablation"):
        """
        Initialize the ablation study.
        
        Args:
            dataset_path: Path to the dataset CSV file
            sample_size: Number of articles to use for evaluation
            use_gpu: Whether to use GPU for embedding models
            output_dir: Base directory for saving all results
        """
        self.dataset_path = dataset_path
        self.sample_size = sample_size
        self.use_gpu = use_gpu
        self.results = []
        self.dataset = None
        self.embedders = {}
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"ablation_study_{timestamp}")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.data_dir = os.path.join(self.output_dir, "data")
        self.reports_dir = os.path.join(self.output_dir, "reports")
        
        # Create all necessary directories
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Initialize components
        self.graph_builder = AblationGraphBuilder()
        self.csv_exporter = CSVResultExporter()
        self.rouge_plot_generator = RougePlotGenerator()
        
        logger.info(f"Output directory created: {self.output_dir}")
    
    def _define_configurations(self) -> List[AblationConfig]:
        """
        Define the ablation study configurations.
        
        Returns:
            List of AblationConfig objects for each experimental condition
        """
        return [
            # Base configuration (optimal from hyperparameter tuning)
            AblationConfig(
                name='Full Model (SBERT)',
                embedder_kind='sbert',
                sim_threshold=0.3,
                use_sent_sent=True,
                use_sent_name=True,
                use_name_name=True,
                description='Complete model with all graph components using SBERT embeddings'
            ),
            # Component ablation: Remove sentence-sentence edges
            AblationConfig(
                name='No Sentence-Sentence Edges',
                embedder_kind='sbert',
                sim_threshold=0.3,
                use_sent_sent=False,
                use_sent_name=True,
                use_name_name=True,
                description='Ablated: Semantic similarity relationships between sentences'
            ),
            # Component ablation: Remove sentence-name edges
            AblationConfig(
                name='No Sentence-Name Edges',
                embedder_kind='sbert',
                sim_threshold=0.3,
                use_sent_sent=True,
                use_sent_name=False,
                use_name_name=True,
                description='Ablated: Connections between sentences and named entities'
            ),
            # Component ablation: Remove name-name edges
            AblationConfig(
                name='No Name-Name Edges',
                embedder_kind='sbert',
                sim_threshold=0.3,
                use_sent_sent=True,
                use_sent_name=True,
                use_name_name=False,
                description='Ablated: Co-occurrence relationships between named entities'
            ),
            # Minimal configuration: Only sentence-sentence edges
            AblationConfig(
                name='Only Sentence-Sentence',
                embedder_kind='sbert',
                sim_threshold=0.3,
                use_sent_sent=True,
                use_sent_name=False,
                use_name_name=False,
                description='Minimal graph with only semantic similarity edges'
            ),
            # Embedding ablation: TF-IDF baseline
            AblationConfig(
                name='TF-IDF Baseline',
                embedder_kind='tfidf',
                sim_threshold=0.1,
                use_sent_sent=True,
                use_sent_name=True,
                use_name_name=True,
                description='Complete model with TF-IDF embeddings instead of SBERT'
            ),
            # Embedding ablation: BOW baseline
            AblationConfig(
                name='BOW Baseline',
                embedder_kind='bow',
                sim_threshold=0.1,
                use_sent_sent=True,
                use_sent_name=True,
                use_name_name=True,
                description='Complete model with Bag-of-Words embeddings'
            ),
            # Extreme ablation: Only names
            AblationConfig(
                name='Only Name Relationships',
                embedder_kind='sbert',
                sim_threshold=0.3,
                use_sent_sent=False,
                use_sent_name=True,
                use_name_name=True,
                description='Graph based solely on named entity relationships'
            )
        ]
    
    def _initialize_embedders(self, configurations: List[AblationConfig]) -> None:
        """Initialize embedding models for all required types."""
        logger.info("Initializing embedding models...")
        
        unique_embedders = set(config.embedder_kind for config in configurations)
        
        for embedder_kind in unique_embedders:
            try:
                if embedder_kind in ["tfidf", "bow"]:
                    model_path = f"models/{embedder_kind}_model.pkl"
                    self.embedders[embedder_kind] = create_embedder(
                        embedder_kind, model_path=model_path
                    )
                    logger.info(f"Initialized {embedder_kind.upper()} embedder from {model_path}")
                else:
                    device = "cuda" if self.use_gpu else "cpu"
                    self.embedders[embedder_kind] = create_embedder(
                        embedder_kind, device=device
                    )
                    logger.info(f"Initialized {embedder_kind.upper()} embedder on {device}")
            except Exception as e:
                logger.error(f"Failed to initialize {embedder_kind} embedder: {e}")
                raise
    
    def run_ablation_study(self) -> List[Dict[str, Any]]:
        """
        Run the complete ablation study across all configurations.
        
        Returns:
            List of dictionaries containing results for each configuration
        """
        logger.info("Starting comprehensive ablation study...")
        logger.info(f"Dataset: {self.dataset_path}")
        logger.info(f"Sample size: {self.sample_size}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Load and prepare data
        self._load_data()
        
        # Define configurations and initialize embedders
        configurations = self._define_configurations()
        self._initialize_embedders(configurations)
        
        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Evaluate each configuration
        total_configs = len(configurations)
        
        for config_idx, config in enumerate(configurations, 1):
            logger.info(f"\n[{config_idx}/{total_configs}] Evaluating: {config.name}")
            logger.info(f"Description: {config.description}")
            
            config_results = self._evaluate_configuration(config, scorer)
            self.results.append(config_results)
            
            # Save intermediate results after each configuration
            self._save_intermediate_results()
        
        logger.info("Ablation study completed successfully!")
        return self.results
    
    def _load_data(self) -> None:
        """Load and prepare the dataset."""
        logger.info("Loading dataset...")
        try:
            self.dataset = pd.read_csv(self.dataset_path)
            self.dataset = self.dataset.head(self.sample_size)
            logger.info(f"Loaded {len(self.dataset)} articles")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _evaluate_configuration(self, config: AblationConfig, scorer) -> Dict[str, Any]:
        """
        Evaluate a single configuration.
        
        Args:
            config: Ablation configuration to evaluate
            scorer: ROUGE scorer instance
            
        Returns:
            Dictionary with aggregated results for this configuration
        """
        config_results = {
            'rouge1': [], 'rouge2': [], 'rougeL': [], 'rouge_l2': [], 'time': []
        }
        
        embedder = self.embedders[config.embedder_kind]
        summary_config = self.graph_builder.build_config(config)
        successful_articles = 0
        
        for i, row in self.dataset.iterrows():
            article = row['article']
            reference = row['highlights']
            
            try:
                start_time = time.time()
                
                # Use the new GraphSummarizer with ablation settings
                summarizer = GraphSummarizer()
                
                # Generate summary using the new summarizer with embedder passed to summarize method
                result = summarizer.summarize(article, summary_config, embedder)
                generated_summary = ' '.join(result.top_sentences)
                generation_time = time.time() - start_time
                
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
                
                successful_articles += 1
                
            except Exception as e:
                logger.warning(f"Error processing article {i}: {str(e)}")
                continue
        
        # Calculate aggregate statistics
        return self._aggregate_config_results(config, config_results, successful_articles)
    
    def _aggregate_config_results(self, config: AblationConfig, 
                                config_results: Dict[str, List], 
                                successful_articles: int) -> Dict[str, Any]:
        """Calculate aggregate statistics for a configuration."""
        # Check if we have any successful articles
        if successful_articles == 0:
            logger.warning(f"No successful articles for configuration: {config.name}")
            return self._create_empty_config_result(config, successful_articles)
        
        return {
            'config_name': config.name,
            'description': config.description,
            'embedder': config.embedder_kind,
            'sim_threshold': config.sim_threshold,
            'use_sent_sent': config.use_sent_sent,
            'use_sent_name': config.use_sent_name,
            'use_name_name': config.use_name_name,
            'avg_rouge1': np.mean(config_results['rouge1']),
            'avg_rouge2': np.mean(config_results['rouge2']),
            'avg_rougeL': np.mean(config_results['rougeL']),
            'avg_rouge_l2': np.mean(config_results['rouge_l2']),
            'std_rouge1': np.std(config_results['rouge1']),
            'std_rouge2': np.std(config_results['rouge2']),
            'std_rougeL': np.std(config_results['rougeL']),
            'std_rouge_l2': np.std(config_results['rouge_l2']),
            'avg_time': np.mean(config_results['time']),
            'std_time': np.std(config_results['time']),
            'n_articles': successful_articles,
            'success_rate': successful_articles / len(self.dataset)
        }
    
    def _create_empty_config_result(self, config: AblationConfig, successful_articles: int) -> Dict[str, Any]:
        """Create a result entry for configurations with no successful articles."""
        return {
            'config_name': config.name,
            'description': config.description,
            'embedder': config.embedder_kind,
            'sim_threshold': config.sim_threshold,
            'use_sent_sent': config.use_sent_sent,
            'use_sent_name': config.use_sent_name,
            'use_name_name': config.use_name_name,
            'avg_rouge1': 0.0,
            'avg_rouge2': 0.0,
            'avg_rougeL': 0.0,
            'avg_rouge_l2': 0.0,
            'std_rouge1': 0.0,
            'std_rouge2': 0.0,
            'std_rougeL': 0.0,
            'std_rouge_l2': 0.0,
            'avg_time': 0.0,
            'std_time': 0.0,
            'n_articles': successful_articles,
            'success_rate': 0.0
        }
    
    def _save_intermediate_results(self) -> None:
        """Save intermediate results to avoid data loss."""
        if self.results:
            df = pd.DataFrame(self.results)
            intermediate_path = os.path.join(self.data_dir, "intermediate_results.csv")
            df.to_csv(intermediate_path, index=False)
            logger.info(f"Intermediate results saved to {intermediate_path}")
    
    def export_all_results(self) -> Dict[str, str]:
        """
        Export all results to organized directory structure.
        
        Returns:
            Dictionary with paths to exported files
        """
        if not self.results:
            logger.warning("No results to export!")
            return {}
        
        df = pd.DataFrame(self.results)
        exported_files = {}
        
        try:
            # Export CSV results
            csv_files = self.csv_exporter.export(self.results, self.data_dir)
            exported_files.update(csv_files)
            
            # Generate and save plots (only if we have successful configurations)
            if not df.empty and df['n_articles'].sum() > 0:
                plot_files = self._export_all_plots()
                exported_files.update(plot_files)
            
            # Generate and save reports
            report_files = self._export_all_reports(df)
            exported_files.update(report_files)
            
            # Create a README file for the export
            self._create_export_readme(exported_files)
            
            logger.info(f"All results successfully exported to: {self.output_dir}")
            return exported_files
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            return {}
    
    def _export_all_plots(self) -> Dict[str, str]:
        """Generate and export all plots in multiple formats."""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        # Filter out configurations with no successful articles
        df = df[df['n_articles'] > 0]
        
        if df.empty:
            logger.warning("No successful configurations to plot")
            return {}
            
        plot_files = {}
        
        try:
            # ROUGE scores plot
            rouge_plot_path = os.path.join(self.plots_dir, "rouge_scores_comparison.png")
            self.rouge_plot_generator.generate_plot(df, rouge_plot_path)
            plot_files['rouge_plot'] = rouge_plot_path
            
            logger.info(f"All plots exported to: {self.plots_dir}")
            return plot_files
            
        except Exception as e:
            logger.error(f"Failed to export plots: {e}")
            return {}
    
    def _export_all_reports(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate and export all reports."""
        report_files = {}
        
        try:
            # JSON report
            json_report_path = os.path.join(self.reports_dir, "ablation_study_report.json")
            report = self._generate_json_report(df)
            with open(json_report_path, 'w') as f:
                json.dump(report, f, indent=2)
            report_files['json_report'] = json_report_path
            
            # Text summary
            text_report_path = os.path.join(self.reports_dir, "ablation_study_summary.txt")
            self._generate_text_report(df, text_report_path)
            report_files['text_report'] = text_report_path
            
            logger.info(f"All reports exported to: {self.reports_dir}")
            return report_files
            
        except Exception as e:
            logger.error(f"Failed to export reports: {e}")
            return {}
    
    def _generate_json_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate JSON report with study findings."""
        if df.empty:
            return {}
        
        # Filter out configurations with no successful articles
        valid_df = df[df['n_articles'] > 0]
        if valid_df.empty:
            return {}
        
        baseline_row = valid_df[valid_df['config_name'] == 'Full Model (SBERT)']
        baseline = baseline_row.iloc[0] if not baseline_row.empty else valid_df.iloc[0]
        
        findings = {
            'best_configuration': valid_df.loc[valid_df['avg_rouge_l2'].idxmax()].to_dict(),
            'worst_configuration': valid_df.loc[valid_df['avg_rouge_l2'].idxmin()].to_dict(),
            'component_importance': {},
            'embedding_comparison': {},
            'performance_summary': {
                'rouge_l2_range': valid_df['avg_rouge_l2'].max() - valid_df['avg_rouge_l2'].min(),
                'avg_processing_time': valid_df['avg_time'].mean(),
                'total_configurations': len(valid_df),
                'success_rate_overall': valid_df['n_articles'].sum() / (len(valid_df) * self.sample_size)
            }
        }
        
        # Component importance analysis
        for component in ['use_sent_sent', 'use_sent_name', 'use_name_name']:
            component_configs = valid_df[valid_df[component] == False]
            if not component_configs.empty:
                avg_drop = baseline['avg_rouge_l2'] - component_configs['avg_rouge_l2'].mean()
                findings['component_importance'][component] = {
                    'performance_drop': avg_drop,
                    'significance': 'high' if avg_drop > 0.05 else 'medium' if avg_drop > 0.02 else 'low'
                }
        
        # Embedding comparison
        embedder_groups = valid_df.groupby('embedder').agg({
            'avg_rouge_l2': 'mean',
            'avg_time': 'mean',
            'n_articles': 'sum'
        }).to_dict('index')
        findings['embedding_comparison'] = embedder_groups
        
        return findings
    
    def _generate_text_report(self, df: pd.DataFrame, output_path: str) -> None:
        """Generate a text summary report."""
        with open(output_path, 'w') as f:
            f.write("ABLATION STUDY SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.dataset_path}\n")
            f.write(f"Sample Size: {self.sample_size}\n\n")
            
            f.write("RESULTS SUMMARY:\n")
            f.write("-" * 30 + "\n")
            for _, row in df.iterrows():
                f.write(f"{row['config_name']}:\n")
                f.write(f"  ROUGE L2: {row['avg_rouge_l2']:.4f} | ")
                f.write(f"Time: {row['avg_time']:.2f}s | ")
                f.write(f"Success: {row['n_articles']}/{self.sample_size}\n")
            
            # Only show key findings if we have successful configurations
            valid_df = df[df['n_articles'] > 0]
            if not valid_df.empty:
                f.write("\nKEY FINDINGS:\n")
                f.write("-" * 30 + "\n")
                best_idx = valid_df['avg_rouge_l2'].idxmax()
                worst_idx = valid_df['avg_rouge_l2'].idxmin()
                f.write(f"Best: {valid_df.loc[best_idx, 'config_name']} (ROUGE L2: {valid_df.loc[best_idx, 'avg_rouge_l2']:.4f})\n")
                f.write(f"Worst: {valid_df.loc[worst_idx, 'config_name']} (ROUGE L2: {valid_df.loc[worst_idx, 'avg_rouge_l2']:.4f})\n")
                f.write(f"Range: {valid_df['avg_rouge_l2'].max() - valid_df['avg_rouge_l2'].min():.4f}\n")
            else:
                f.write("\nNo successful configurations to analyze.\n")
    
    def _create_export_readme(self, exported_files: Dict[str, str]) -> None:
        """Create a README file explaining the exported files."""
        readme_path = os.path.join(self.output_dir, "README.md")
        
        readme_content = f"""
# Ablation Study Results

This directory contains the complete results of the ablation study conducted on {datetime.now().strftime("%Y-%m-%d")}.

## Directory Structure

- `data/`: Contains CSV files with detailed results
- `plots/`: Contains visualization plots in PNG and PDF formats
- `reports/`: Contains comprehensive reports in JSON, HTML, and text formats

## Study Parameters
- Dataset: {self.dataset_path}
- Sample Size: {self.sample_size}
- Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Usage
The results can be used to:
1. Understand the contribution of different graph components
2. Compare embedding methods
3. Identify optimal configurations
4. Guide future model improvements
"""
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
    
    def plot_and_export_results(self, show_plots: bool = True) -> Dict[str, str]:
        """
        Generate plots and export all results.
        
        Args:
            show_plots: Whether to display plots interactively
            
        Returns:
            Dictionary with paths to all exported files
        """
        if not self.results:
            logger.error("No results to plot and export!")
            return {}
            
        df = pd.DataFrame(self.results)
        
        # Print summary
        self._print_summary_table(df)
        
        # Export all results
        exported_files = self.export_all_results()
        
        return exported_files
    
    def _print_summary_table(self, df: pd.DataFrame) -> None:
        """Print a formatted summary table of results."""
        print("\n" + "="*120)
        print("COMPREHENSIVE ABLATION STUDY SUMMARY")
        print("="*120)
        
        summary_df = df[['config_name', 'embedder', 'avg_rouge1', 'avg_rouge2', 
                        'avg_rougeL', 'avg_rouge_l2', 'avg_time', 'n_articles']].copy()
        summary_df.columns = ['Configuration', 'Embedder', 'ROUGE-1', 'ROUGE-2', 
                             'ROUGE-L', 'ROUGE L2-Norm', 'Time (s)', 'Success Count']
        summary_df = summary_df.round(4)
        
        print(summary_df.to_string(index=False, float_format='%.4f'))
        print("="*120)
        
        # Print key insights only if we have successful configurations
        valid_df = df[df['n_articles'] > 0]
        if not valid_df.empty:
            best_idx = valid_df['avg_rouge_l2'].idxmax()
            worst_idx = valid_df['avg_rouge_l2'].idxmin()
            
            print(f"\nKEY INSIGHTS:")
            print(f"• Best configuration: {valid_df.loc[best_idx, 'config_name']} "
                  f"(ROUGE L2-Norm: {valid_df.loc[best_idx, 'avg_rouge_l2']:.4f})")
            print(f"• Worst configuration: {valid_df.loc[worst_idx, 'config_name']} "
                  f"(ROUGE L2-Norm: {valid_df.loc[worst_idx, 'avg_rouge_l2']:.4f})")
            print(f"• Performance range: {valid_df['avg_rouge_l2'].max() - valid_df['avg_rouge_l2'].min():.4f}")
            print(f"• Average processing time: {valid_df['avg_time'].mean():.2f} seconds")
        else:
            print(f"\nKEY INSIGHTS:")
            print(f"• No successful configurations to analyze")
        
        print("="*120)


if __name__ == "__main__":
    # Run comprehensive ablation study with export
    ablation = AblationStudy(
        dataset_path="Datasets/cnn_dailymail/validation.csv",
        sample_size=100,  # Reduced for faster testing
        use_gpu=True,
        output_dir="results/ablation"
    )
    
    try:
        # Run the study
        results = ablation.run_ablation_study()
        
        # Plot and export all results
        exported_files = ablation.plot_and_export_results(show_plots=False)
        
        logger.info("Ablation study completed successfully!")
        logger.info(f"All results exported to: {ablation.output_dir}")
        
        # Print export summary
        if exported_files:
            print("\n" + "="*80)
            print("EXPORT SUMMARY")
            print("="*80)
            for file_type, file_path in exported_files.items():
                print(f"✓ {file_type}: {file_path}")
            print("="*80)
        
    except Exception as e:
        logger.error(f"Ablation study failed: {e}")
        raise