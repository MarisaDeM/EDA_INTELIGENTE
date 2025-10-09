"""
Agent module for EDA Intelligent Agent
Extracted and adapted from Jupyter notebook for production deployment
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import google.generativeai as genai
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

# Imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image

warnings.filterwarnings('ignore')

# Directory to save temporary plots
temp_dir = "temp_plots"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)


class CSVAnalysisAgent:
    """AI Agent specialized in Exploratory Data Analysis (EDA) for CSV files."""
    
    def __init__(self, api_key: str):
        """Initialize the agent with Gemini API key."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.memory: List[Dict[str, Any]] = []
        self.current_data: Optional[pd.DataFrame] = None
        self.data_summary: Optional[Dict[str, Any]] = None
        self.generated_image_bytes: Dict[str, bytes] = {}  # Store image bytes for PDF
        self.generated_figures: Dict[str, go.Figure] = {}  # Store Figure objects
    
    def _clear_temp_plots(self):
        """Clear temporary plots directory to avoid file accumulation."""
        if os.path.exists(temp_dir):
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        self.generated_image_bytes.clear()
        self.generated_figures.clear()
    
    def load_csv_file(self, file_path: str) -> str:
        """Load a CSV file for analysis and clear previous analyses."""
        self._clear_temp_plots()  # Clear old plots when loading new file
        self.memory = []  # Reset memory
        try:
            self.current_data = pd.read_csv(file_path)
            self.data_summary = self._generate_data_summary()
            memory_entry = {
                'action': 'load_data',
                'timestamp': pd.Timestamp.now(),
                'data_shape': self.current_data.shape,
                'columns': list(self.current_data.columns)
            }
            self.memory.append(memory_entry)
            return f"Data loaded successfully! Dimension: {self.current_data.shape}"
        except Exception as e:
            self.current_data = None
            self.data_summary = None
            return f"Error loading file: {str(e)}"
    
    def _generate_data_summary(self) -> Optional[Dict[str, Any]]:
        """Generate summary of loaded data."""
        if self.current_data is None:
            return None
        return {
            'shape': self.current_data.shape,
            'columns': list(self.current_data.columns),
            'dtypes': self.current_data.dtypes.astype(str).to_dict(),
            'missing_values': self.current_data.isnull().sum().to_dict(),
            'numeric_columns': list(self.current_data.select_dtypes(include=np.number).columns),
            'categorical_columns': list(self.current_data.select_dtypes(include='object').columns)
        }
    
    def analyze_data_types(self) -> Tuple[str, Optional[go.Figure]]:
        """Analyze data types and return text analysis with visualization."""
        if self.current_data is None or self.data_summary is None:
            return "No data loaded.", None
        
        analysis = {
            'Numeric': len(self.data_summary['numeric_columns']),
            'Categorical': len(self.data_summary['categorical_columns']),
            'Total Columns': len(self.data_summary['columns'])
        }
        
        fig = px.bar(x=list(analysis.keys()), y=list(analysis.values()), 
                    title="Data Types Distribution")
        self.generated_figures['datatypes_plot'] = fig
        
        # Try to generate image bytes for PDF
        img_bytes = None
        try:
            img_bytes = fig.to_image(format="png")
            self.generated_image_bytes['datatypes_plot'] = img_bytes
            print("Image bytes for datatypes_plot generated and stored (if kaleido works).")
        except Exception as e:
            print(f"Error generating image bytes for datatypes_plot (kaleido): {e}")
        
        self.memory.append({
            'action': 'analyze_data_types',
            'timestamp': pd.Timestamp.now(),
            'analysis': analysis,
            'image_key': 'datatypes_plot' if img_bytes else None
        })
        
        return str(analysis), fig
    
    def statistical_summary(self) -> str:
        """Generate statistical summary of the data."""
        if self.current_data is None or self.data_summary is None:
            return "No data loaded."
        
        numeric_stats = self.current_data.describe().to_string()
        categorical_stats = {
            col: {
                'unique_values': int(self.current_data[col].nunique()),
                'most_frequent': str(self.current_data[col].mode().iloc[0])
            }
            for col in self.data_summary['categorical_columns']
        }
        
        summary_text = f"Numeric Statistics:\n{numeric_stats}\n\nCategorical Statistics:\n{categorical_stats}"
        
        self.memory.append({
            'action': 'statistical_summary',
            'timestamp': pd.Timestamp.now(),
            'summary': summary_text
        })
        
        return summary_text
    
    def detect_outliers(self, column_name: Optional[str] = None) -> Tuple[str, Optional[go.Figure]]:
        """Detect outliers using IQR method and return boxplots."""
        if self.current_data is None or self.data_summary is None:
            return "No data loaded.", None
        
        numeric_cols = self.data_summary['numeric_columns']
        cols_to_analyze = [column_name] if column_name and column_name in numeric_cols else numeric_cols[:4]
        
        if not cols_to_analyze:
            return "No numeric columns to analyze.", None
        
        fig = make_subplots(rows=1, cols=len(cols_to_analyze), 
                           subplot_titles=cols_to_analyze)
        
        outliers_info = {}
        for i, col in enumerate(cols_to_analyze):
            series = self.current_data[col].dropna()
            if not series.empty:
                Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                outliers = self.current_data[(self.current_data[col] < lower) | 
                                           (self.current_data[col] > upper)]
                outliers_info[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(self.current_data)) * 100
                }
                fig.add_trace(go.Box(y=self.current_data[col], name=col), 
                             row=1, col=i+1)
        
        fig.update_layout(title="Outlier Detection - Boxplots")
        info_text = f"Outlier Information:\n{outliers_info}"
        
        self.generated_figures['outliers_plot'] = fig
        
        # Try to generate image bytes for PDF
        img_bytes = None
        try:
            img_bytes = fig.to_image(format="png")
            self.generated_image_bytes['outliers_plot'] = img_bytes
            print("Image bytes for outliers_plot generated and stored (if kaleido works).")
        except Exception as e:
            print(f"Error generating image bytes for outliers_plot (kaleido): {e}")
        
        self.memory.append({
            'action': 'detect_outliers',
            'timestamp': pd.Timestamp.now(),
            'outliers_info': outliers_info,
            'image_key': 'outliers_plot' if img_bytes else None
        })
        
        return info_text, fig
    
    def correlation_analysis(self) -> Tuple[str, Optional[go.Figure]]:
        """Perform correlation analysis on numeric data."""
        if self.current_data is None:
            return "No data loaded.", None
        
        numeric_data = self.current_data.select_dtypes(include=np.number)
        if numeric_data.shape[1] < 2:
            return "Too few numeric columns for correlation.", None
        
        corr_matrix = numeric_data.corr(numeric_only=True)
        fig = px.imshow(corr_matrix, title="Correlation Matrix", 
                       color_continuous_scale="RdBu", aspect="auto")
        
        # Find high correlations
        high_corr = [f"{c1}-{c2}: {corr_matrix.loc[c1, c2]:.3f}" 
                    for c1 in corr_matrix.columns for c2 in corr_matrix.columns 
                    if c1 != c2 and abs(corr_matrix.loc[c1, c2]) > 0.7]
        
        info_text = f"High Correlations (|r| > 0.7):\n{high_corr}"
        
        self.generated_figures['correlation_plot'] = fig
        
        # Try to generate image bytes for PDF
        img_bytes = None
        try:
            img_bytes = fig.to_image(format="png")
            self.generated_image_bytes['correlation_plot'] = img_bytes
            print("Image bytes for correlation_plot generated and stored (if kaleido works).")
        except Exception as e:
            print(f"Error generating image bytes for correlation_plot (kaleido): {e}")
        
        self.memory.append({
            'action': 'correlation_analysis',
            'timestamp': pd.Timestamp.now(),
            'high_correlations': high_corr,
            'image_key': 'correlation_plot' if img_bytes else None
        })
        
        return info_text, fig
    
    def save_plots_as_html(self) -> List[str]:
        """Save generated plots as HTML files in temporary directory."""
        saved_files = []
        print(f"Trying to save {len(self.generated_figures)} figures as individual HTML...")
        
        if not self.generated_figures:
            print("No figures generated to save as HTML.")
            return []
        
        for key, fig in self.generated_figures.items():
            html_filename = os.path.join(temp_dir, f"{key}.html")
            try:
                fig.write_html(html_filename)
                saved_files.append(html_filename)
                print(f"Figure {key} saved as HTML at {html_filename}")
            except Exception as e:
                print(f"Error saving figure {key} as HTML: {e}")
        
        print(f"Completed saving individual plots. Total files saved: {len(saved_files)}")
        return saved_files
    
    def _format_memory(self) -> str:
        """Format memory for context in conclusions."""
        if not self.memory:
            return "No previous analysis."
        
        formatted = []
        for entry in self.memory[-10:]:  # Last 10 entries
            action = entry.get('action', '').replace('_', ' ').title()
            img_info = f" (Image Key: {entry['image_key']})" if entry.get('image_key') else ""
            if action == "Query":
                response_snippet = entry['response'][:150] + "..." if len(entry['response']) > 150 else entry['response']
                formatted.append(f"- Question: {entry['question']} - Response: {response_snippet}")
            else:
                formatted.append(f"- Analysis: {action}{img_info}")
        
        return "\n".join(formatted)
    
    def get_conclusions(self) -> str:
        """Generate conclusions using AI based on all analyses performed."""
        if self.current_data is None or self.data_summary is None:
            return "No data loaded."
        
        context = f"""Based on all analyses performed on the dataset (historical in memory), 
        generate comprehensive conclusions. Include references to image file names of generated 
        plots when relevant (use the Image Keys from history if available).
        
        Analysis History:
        {self._format_memory()}
        
        Provide conclusions in Portuguese with clear topics:
        1. Key insights discovered.
        2. Patterns and trends identified.
        3. Data quality assessment (missing values, outliers).
        4. Recommendations for future analyses or business actions."""
        
        try:
            response = self.model.generate_content(context)
            return response.text if hasattr(response, 'text') and response.text else "The model did not generate conclusions."
        except Exception as e:
            return f"Error generating conclusions: {str(e)}"
    
    def generate_pdf_report(self) -> Optional[str]:
        """Generate a PDF report with conclusions and charts."""
        if self.current_data is None or not self.memory:
            return None
        
        pdf_filename = "relatorio_eda.pdf"
        doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Add title
        story.append(Paragraph("Relatório de Análise Exploratória de Dados (EDA)", styles['h1']))
        story.append(Spacer(1, 12))
        
        # Add conclusions
        conclusions = self.get_conclusions()
        if conclusions and conclusions != "No data loaded." and not conclusions.startswith("Error generating conclusions"):
            story.append(Paragraph("Conclusões e Insights", styles['h2']))
            for line in conclusions.split('\n'):
                if line.strip():
                    story.append(Paragraph(line, styles['Normal']))
                    story.append(Spacer(1, 6))
            story.append(Spacer(1, 12))
        
        # Add charts if available in bytes
        if self.generated_image_bytes and any(self.generated_image_bytes.values()):
            story.append(Paragraph("Gráficos Gerados", styles['h2']))
            story.append(Spacer(1, 12))
            
            print(f"Trying to add {len([b for b in self.generated_image_bytes.values() if b is not None])} images to PDF...")
            
            for key, img_bytes in self.generated_image_bytes.items():
                if img_bytes:
                    try:
                        print(f"Adding image {key} to PDF...")
                        img_reader = ImageReader(img_bytes)
                        img = Image(img_reader)
                        
                        # Resize image to fit page
                        img_width, img_height = img.drawWidth, img.drawHeight
                        page_width, page_height = letter
                        max_width = page_width - 2 * doc.leftMargin
                        max_height = page_height - 2 * doc.bottomMargin
                        
                        aspect = img_height / float(img_width)
                        if img_width > max_width:
                            img_width = max_width
                            img_height = img_width * aspect
                        if img_height > max_height:
                            img_height = max_height
                            img_width = img_height / aspect
                        
                        img.drawWidth = img_width
                        img.drawHeight = img_height
                        
                        story.append(Paragraph(key.replace('_', ' ').title(), styles['h3']))
                        story.append(Spacer(1, 6))
                        story.append(img)
                        story.append(Spacer(1, 12))
                        
                        print(f"Image {key} added successfully to story.")
                    except Exception as e:
                        print(f"Error adding image {key} to PDF: {e}")
                        story.append(Paragraph(f"Error loading image {key}.", styles['Normal']))
                        story.append(Spacer(1, 6))
                else:
                    print(f"Skipping image {key} - bytes not generated (export error).")
                    story.append(Paragraph(f"Image {key} not generated (export error).", styles['Normal']))
                    story.append(Spacer(1, 6))
        elif self.current_data is not None:
            story.append(Paragraph("Gráficos Gerados", styles['h2']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("Could not include charts in PDF report due to technical issues with image export.", styles['Normal']))
            story.append(Spacer(1, 12))
        
        try:
            doc.build(story)
            print(f"PDF report {pdf_filename} generated successfully.")
            return pdf_filename
        except Exception as e:
            print(f"Error building PDF: {e}")
            return None
    
    def generate_html_report(self) -> Optional[str]:
        """Generate an HTML report with conclusions and embedded graphs."""
        if self.current_data is None or not self.memory:
            print("No data loaded or no analysis in memory to generate HTML report.")
            return None
        
        html_filename = os.path.join(temp_dir, "relatorio_eda.html")
        conclusions = self.get_conclusions()
        
        # Generate HTML for charts
        chart_html = ""
        print(f"Checking generated figures to include in HTML: {len(self.generated_figures)}")
        
        if self.generated_figures:
            for key, fig in self.generated_figures.items():
                try:
                    print(f"Converting chart {key} to HTML...")
                    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                    chart_html += f"""
                    <div class="graph">
                        <h3>{key.replace('_', ' ').title()}</h3>
                        {fig_html}
                    </div>
                    """
                    print(f"Chart {key} converted to HTML and included.")
                except Exception as e:
                    print(f"Error converting chart {key} to HTML: {e}")
                    chart_html += f"""
                    <div class="graph">
                        <h3>{key.replace('_', ' ').title()}</h3>
                        <p>Could not render chart {key}.</p>
                    </div>
                    """
        else:
            print("No figures generated to include in HTML report.")
            chart_html = "<p>No charts generated to include in this report.</p>"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Relatório de Análise Exploratória de Dados (EDA)</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .section {{ margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid #eee; }}
                .graph {{ margin-top: 15px; text-align: center; }}
            </style>
        </head>
        <body>
            <h1>Relatório de Análise Exploratória de Dados (EDA)</h1>
            
            <div class="section">
                <h2>Conclusões e Insights</h2>
                {"".join([f"<p>{line}</p>" for line in conclusions.split('\n') if line.strip()]) if conclusions and conclusions != "No data loaded." and not conclusions.startswith("Error generating conclusions") else "<p>No conclusions generated.</p>"}
            </div>
            
            <div class="section">
                <h2>Gráficos Gerados</h2>
                {chart_html}
            </div>
        </body>
        </html>
        """
        
        try:
            with open(html_filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"HTML report {html_filename} generated successfully.")
            return html_filename
        except Exception as e:
            print(f"Error generating HTML report: {e}")
            return None