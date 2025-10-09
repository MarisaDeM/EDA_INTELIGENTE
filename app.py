"""
Main application file for EDA Intelligent Agent
Gradio interface adapted from notebook for production deployment
"""

import os
import gradio as gr
from agent import CSVAnalysisAgent

def create_gradio_interface(agent=None):
    """Create and configure Gradio interface for the EDA agent."""
    latest_conclusions = ""
    
    def initialize_agent() -> str:
        nonlocal agent
        try:
            # Get API key from environment variable instead of Colab userdata
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                return "❌ GEMINI_API_KEY not found in environment variables. Please configure it."
            
            agent = CSVAnalysisAgent(api_key=api_key)
            return "✅ Agent initialized successfully! Ready to load data."
        except Exception as e:
            return f"❌ Error initializing agent: {e}. Check API Key."
    
    def load_file(file: gr.File) -> str:
        nonlocal agent
        if agent is None:
            return "❌ Initialize the agent first!"
        
        if file is None:
            return "❌ No file selected."
        
        status = agent.load_csv_file(file.name)
        return status
    
    def update_images_list(analysis_text, fig):
        """Simplified return for text and figure."""
        return analysis_text, fig
    
    # Interface handler functions
    def get_data_types_interface():
        if agent is None:
            return "❌ Initialize the agent first!", None
        analysis_text, fig = agent.analyze_data_types()
        return update_images_list(analysis_text, fig)
    
    def detect_outliers_interface():
        if agent is None:
            return "❌ Initialize the agent first!", None
        analysis_text, fig = agent.detect_outliers()
        return update_images_list(analysis_text, fig)
    
    def correlation_analysis_interface():
        if agent is None:
            return "❌ Initialize the agent first!", None
        analysis_text, fig = agent.correlation_analysis()
        return update_images_list(analysis_text, fig)
    
    def generate_and_download_pdf_interface():
        nonlocal agent
        if agent is None:
            return None
        pdf_path = agent.generate_pdf_report()
        return pdf_path
    
    def generate_and_download_html_report_interface():
        nonlocal agent
        if agent is None:
            return None
        html_path = agent.generate_html_report()
        return html_path
    
    def save_plots_as_html_interface():
        nonlocal agent
        if agent is None:
            return None
        saved_files = agent.save_plots_as_html()
        if saved_files:
            return gr.File(value=saved_files, visible=True, label="Download Individual HTML Plots")
        else:
            return None
    
    def get_conclusions_interface() -> str:
        nonlocal agent, latest_conclusions
        if agent is None:
            return "❌ Initialize the agent first!"
        latest_conclusions = agent.get_conclusions()
        return latest_conclusions
    
    # Create Gradio interface
    with gr.Blocks(title="EDA Agent", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 📊 Agente Inteligente para Análise Exploratória de Dados (EDA)")
        
        with gr.Row():
            with gr.Column():
                init_btn = gr.Button("🚀 Initialize Agent", variant="primary")
                init_output = gr.Textbox(label="Initialization Status", interactive=False)
                
                file_input = gr.File(label="📁 Upload CSV file", file_types=[".csv"])
                load_output = gr.Textbox(label="Loading Status", interactive=False)
        
        with gr.Tabs():
            with gr.TabItem("📊 Automated Analysis"):
                with gr.Row():
                    types_btn = gr.Button("🔍 Data Types")
                    stats_btn = gr.Button("📈 Statistical Summary")
                    outliers_btn = gr.Button("⚠️ Detect Outliers")
                    corr_btn = gr.Button("🔗 Correlation")
                
                analysis_output = gr.Textbox(label="Analysis Results", lines=8, interactive=False)
                analysis_plot = gr.Plot(label="Analysis Visualization")
            
            with gr.TabItem("🤖 Conclusions and Download"):
                conclusions_btn = gr.Button("✨ Generate Final Conclusions", variant="secondary")
                conclusions_output = gr.Textbox(label="Conclusions and Insights", lines=15, interactive=False)
                
                gr.Markdown("### 📄 Complete PDF Report")
                with gr.Row():
                    generate_pdf_btn = gr.Button("📄 Generate PDF Report", variant="secondary")
                    download_pdf_file_output = gr.File(label="EDA Report .pdf", interactive=False)
                
                gr.Markdown("### 🌐 Complete HTML Report")
                with gr.Row():
                    generate_html_report_btn = gr.Button("🌐 Generate HTML Report", variant="secondary")
                    download_html_report_output = gr.File(label="EDA Report .html", interactive=False)
                
                gr.Markdown("### 📊 Save Individual HTML Plots")
                with gr.Row():
                    save_html_btn = gr.Button("💾 Save Individual HTML Plots", variant="secondary")
                    download_individual_html_output = gr.File(label="Download Individual HTML Plots", 
                                                            interactive=False, visible=False)
        
        # Event Handlers
        init_btn.click(fn=initialize_agent, outputs=[init_output])
        
        file_input.change(fn=load_file, inputs=[file_input], outputs=[load_output])
        
        stats_btn.click(fn=lambda: agent.statistical_summary() if agent else "Initialize agent first!", 
                       outputs=[analysis_output])
        
        types_btn.click(fn=get_data_types_interface, outputs=[analysis_output, analysis_plot])
        outliers_btn.click(fn=detect_outliers_interface, outputs=[analysis_output, analysis_plot])
        corr_btn.click(fn=correlation_analysis_interface, outputs=[analysis_output, analysis_plot])
        
        conclusions_btn.click(fn=get_conclusions_interface, outputs=[conclusions_output])
        
        generate_pdf_btn.click(fn=generate_and_download_pdf_interface, 
                             outputs=[download_pdf_file_output])
        
        generate_html_report_btn.click(fn=generate_and_download_html_report_interface, 
                                     outputs=[download_html_report_output])
        
        save_html_btn.click(fn=save_plots_as_html_interface, 
                           outputs=[download_individual_html_output])
    
    return interface


def main():
    """Main application entry point."""
    print("🚀 Starting EDA Intelligent Agent...")
    print("📋 Please wait for the Gradio interface to load...")
    
    # Create interface without pre-initialized agent
    interface = create_gradio_interface()
    
    # Launch interface
    port = int(os.environ.get("PORT", 7860))
    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False  # Don't create temporary share links in production
    )


if __name__ == "__main__":
    main()