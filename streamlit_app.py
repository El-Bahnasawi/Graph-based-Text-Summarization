# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
from io import StringIO
import sys
import os
import warnings
import networkx as nx


# Suppress warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import refactored summarization system
from summarizer import GraphSummarizer, SummaryConfig
from sent_embd import create_embedder
from reactive_graph import ReactiveGraph

# Import visualization function (keep this in app.py as it's UI-specific)
from app_visualization import create_graph_visualization, create_demo_graph, create_score_comparison_chart

# Page configuration
st.set_page_config(
    page_title="Graph-Based Text Summarization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def setup_sidebar():
    """Configure sidebar controls"""
    st.sidebar.header("⚙️ Configuration")
    
    embedder_kind = st.sidebar.selectbox(
        "Embedding Method",
        ["sbert", "tfidf", "bow", "bert"],
        index=0,
        help="Choose the sentence embedding method"
    )
    
    st.sidebar.subheader("Graph Components")
    use_sentence_edges = st.sidebar.checkbox("Sentence-Sentence Edges", value=True)
    use_sent_name_edges = st.sidebar.checkbox("Sentence-Entity Edges", value=True)
    use_name_name_edges = st.sidebar.checkbox("Entity-Entity Relations", value=True)
    
    st.sidebar.subheader("Parameters")
    sim_threshold = st.sidebar.slider("Similarity Threshold", 0.1, 0.9, 0.3, 0.1)
    num_sentences = st.sidebar.slider("Summary Length", 1, 10, 3)
    damping_factor = st.sidebar.slider("PageRank Damping", 0.5, 0.95, 0.85, 0.05)
    
    return {
        'embedder_kind': embedder_kind,
        'use_sentence_edges': use_sentence_edges,
        'use_sent_name_edges': use_sent_name_edges,
        'use_name_name_edges': use_name_name_edges,
        'sim_threshold': sim_threshold,
        'num_sentences': num_sentences,
        'damping_factor': damping_factor
    }

def get_sample_texts():
    """Return sample texts for demonstration with realistic content suitable for summarization"""
    return {
        "Technology": """Artificial intelligence is transforming various industries by enabling machines to perform tasks that typically require human intelligence.  Recent advancements in deep learning and neural networks have led to breakthroughs in natural language processing, computer vision, and robotics. Companies are increasingly adopting AI to automate processes, enhance decision-making, and create new products and services. However, the rapid development of AI also raises important ethical considerations regarding privacy, bias, and job displacement. Researchers are working on developing explainable AI systems that can be trusted and understood by humans. The future of AI promises even more sophisticated applications, from personalized medicine to autonomous transportation systems. """,
        "Science": """Climate change represents one of the most pressing challenges facing our planet today. Scientists have documented rising global temperatures, melting polar ice caps, and increasing frequency of extreme weather events. The primary driver of climate change is the emission of greenhouse gases from human activities such as burning fossil fuels and deforestation. International agreements like the Paris Accord aim to limit global warming to well below 2 degrees Celsius. Renewable energy sources like solar and wind power are becoming increasingly cost-competitive with traditional fossil fuels. Individual actions, combined with systemic changes in energy production and consumption, are essential for mitigating the worst impacts of climate change. """,
        "Business": """The global economy is experiencing significant shifts due to digital transformation and changing consumer behaviors. E-commerce has revolutionized retail, with online sales growing at an unprecedented rate during the pandemic. Remote work has become mainstream, forcing companies to rethink their office spaces and collaboration tools. Supply chain disruptions have highlighted the importance of resilience and diversification in business operations. Sustainable business practices are no longer optional but essential for long-term success and investor confidence. Companies that embrace innovation and adaptability are best positioned to thrive in this rapidly changing economic landscape. """,
        "Health": """The COVID-19 pandemic has fundamentally changed global healthcare systems and public health approaches. Vaccine development achieved unprecedented speed through mRNA technology and international collaboration. Telemedicine has become widely adopted, providing access to healthcare services remotely. Mental health has gained increased attention as people cope with pandemic-related stress and isolation. Healthcare disparities have been highlighted, emphasizing the need for more equitable access to medical services. Future pandemic preparedness requires robust public health infrastructure and global cooperation.""",
        "Education": """Digital learning technologies have transformed education delivery methods worldwide. Online platforms enable students to access quality education regardless of geographical location. Personalized learning algorithms adapt to individual student needs and learning styles. Traditional assessment methods are being supplemented with continuous evaluation and project-based learning. The digital divide remains a challenge, with unequal access to technology and internet connectivity. Educators are developing hybrid models that combine the best of in-person and online learning experiences. """,
        "Environment": """Biodiversity loss and ecosystem degradation threaten the planet's life support systems. Deforestation in the Amazon and other critical regions continues at alarming rates. Plastic pollution in oceans harms marine life and enters the human food chain. Conservation efforts are showing success in some areas through protected areas and species recovery programs. Sustainable agriculture practices can help feed growing populations while protecting natural resources. Individual consumer choices and corporate responsibility both play crucial roles in environmental protection."""
    }

def display_results(result, original_text):
    """Display summarization results"""
    st.success(f"✅ Summary generated in {result.processing_time:.2f} seconds!")
    
    # Metrics
    col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
    with col_metrics1:
        st.markdown(f'<div class="metric-highlight">⏱️ {result.processing_time:.2f}s</div>', unsafe_allow_html=True)
    with col_metrics2:
        st.markdown(f'<div class="metric-highlight">📝 {len(result.top_sentences)} sentences</div>', unsafe_allow_html=True)
    with col_metrics3:
        st.markdown(f'<div class="metric-highlight">🕸️ {result.graph.nx.number_of_nodes()} nodes</div>', unsafe_allow_html=True)
    with col_metrics4:
        st.markdown(f'<div class="metric-highlight">🔗 {result.graph.nx.number_of_edges()} edges</div>', unsafe_allow_html=True)
    
    # Results in columns
    col_summary, col_graph = st.columns([1, 1])
    
    with col_summary:
        st.subheader("📄 Generated Summary")
        for i, sentence in enumerate(result.top_sentences, 1):
            st.write(f"**{i}.** {sentence}")
        
        with st.expander("📈 Detailed Analysis"):
            original_words = len(original_text.split())
            summary_words = len(' '.join(result.top_sentences).split())
            compression_ratio = (summary_words / original_words) * 100
            
            st.write(f"**Text Statistics:**")
            st.write(f"- Original text: {original_words} words")
            st.write(f"- Summary: {summary_words} words") 
            st.write(f"- Compression: {compression_ratio:.1f}%")
            
            # Show top sentences with FINAL scores
            st.write("**Top sentences by PageRank score:**")
            top_scored = sorted(result.sentence_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            for i, (idx, score) in enumerate(top_scored, 1):
                st.write(f"{i}. **({score:.3f})** Sentence {idx + 1}")
    
    with col_graph:
        st.subheader("🕸️ Knowledge Graph")
        if result.graph.nx.number_of_nodes() > 0:
            # Create and display the graph visualization with FINAL scores
            try:
                fig = create_graph_visualization(result.graph, result.sentence_scores)
                st.pyplot(fig)
                plt.close(fig)
                
                # Optional: Show score comparison
                with st.expander("📊 Score Evolution"):
                    # Calculate initial scores (1/n)
                    n = len(result.sentence_scores)
                    initial_scores = {i: 1.0/n for i in result.sentence_scores.keys()}
                    
                    fig_comparison = create_score_comparison_chart(initial_scores, result.sentence_scores)
                    if fig_comparison:
                        st.pyplot(fig_comparison)
                        plt.close(fig_comparison)
                    
            except Exception as e:
                st.warning(f"Could not generate graph visualization: {str(e)}")
                # Fallback: show graph statistics
                st.write("**Graph Statistics:**")
                st.write(f"- Nodes: {result.graph.nx.number_of_nodes()}")
                st.write(f"- Edges: {result.graph.nx.number_of_edges()}")
                st.write(f"- Density: {nx.density(result.graph.nx):.3f}")
        else:
            st.info("No graph data available")

def main():
    """Main Streamlit application"""
    st.markdown('<h1 class="main-header">🧠 Graph-Based Text Summarization</h1>', unsafe_allow_html=True)
    
    # Setup sidebar
    params = setup_sidebar()
    
    # Text input
    st.subheader("📝 Input Text")
    input_option = st.radio("Input Source", ["Sample Text", "Custom Text", "File Upload"], horizontal=True)
    
    text_input = ""
    sample_texts = get_sample_texts()
    
    if input_option == "Sample Text":
        selected_sample = st.selectbox("Choose sample topic", list(sample_texts.keys()))
        text_input = sample_texts[selected_sample]
        st.text_area("Sample Text", text_input, height=200, key="sample_text")
    elif input_option == "Custom Text":
        text_input = st.text_area("Enter your text here", height=200, 
                                placeholder="Paste or type your text here...")
    else:
        uploaded_file = st.file_uploader("Upload text file", type=['txt'])
        if uploaded_file is not None:
            text_input = uploaded_file.getvalue().decode("utf-8")
            st.text_area("Uploaded Text", text_input, height=200, key="uploaded_text")
    
    # Process button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_clicked = st.button("🚀 Generate Summary", type="primary", use_container_width=True)
    
    # Processing
    if process_clicked and text_input.strip():
        with st.spinner("🔄 Processing text and building graph..."):
            start_time = time.time()
            
            try:
                # Create configuration
                config = SummaryConfig(
                    k=params['num_sentences'],
                    sim_threshold=params['sim_threshold'],
                    embedder_kind=params['embedder_kind'],
                    use_sentence_edges=params['use_sentence_edges'],
                    use_sent_name_edges=params['use_sent_name_edges'],
                    use_name_name_edges=params['use_name_name_edges'],
                    damping=params['damping_factor']
                )
                
                # Initialize services
                embedder = create_embedder(params['embedder_kind'])
                summarizer = GraphSummarizer()
                
                # Generate summary
                result = summarizer.summarize(text_input, config, embedder)
                
                # Display results
                display_results(result, text_input)
                
            except Exception as e:
                st.error(f"❌ Error processing text: {str(e)}")
                st.info("Try adjusting the parameters or using a different embedding method.")
    
    else:
        # Show demo when no processing
        show_demo()

def show_demo():
    """Show demonstration content"""
    st.subheader("🎯 How It Works")
    
    col_info, col_demo = st.columns([2, 1])
    
    with col_info:
        st.markdown("""
        This system uses **graph-based algorithms** to extract important sentences from text:
        
        1. **Text Processing**: Input text is split into sentences
        2. **Embedding**: Sentences are converted to numerical vectors
        3. **Graph Construction**: A knowledge graph is built with different edge types
        4. **PageRank**: Importance scores are computed for each node
        5. **Summary Extraction**: Top-ranked sentences form the summary
        """)
    
    with col_demo:
        st.subheader("🕸️ Sample Graph")
        try:
            demo_graph = create_demo_graph()
            fig = create_graph_visualization(demo_graph)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.info("Graph visualization will appear here after processing")

if __name__ == "__main__":
    main()