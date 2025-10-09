---
title: Agente EDA Inteligente
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 📊 Agente EDA Inteligente

Ferramenta de análise exploratória de dados automatizada com inteligência artificial.

## 🚀 Funcionalidades

- **📁 Upload de CSV**: Carregamento direto de arquivos de dados
- **🔍 Análise Automática**: Tipos de dados, estatísticas e metadados
- **📊 Visualizações**: Gráficos interativos com Plotly
- **⚠️ Detecção de Outliers**: Identificação automática usando método IQR
- **🔗 Correlações**: Matrix de correlação com heatmap
- **🤖 Insights com IA**: Conclusões geradas pelo Google Gemini
- **📄 Relatórios**: Exportação em PDF e HTML

## 🛠️ Tecnologias

- **Frontend**: Gradio
- **Backend**: Python, Pandas, NumPy
- **Visualização**: Plotly, Matplotlib, Seaborn
- **IA**: Google Gemini 2.0 Flash
- **Relatórios**: ReportLab, HTML

## 📖 Como Usar

1. **Initialize**: Clique em "Initialize Agent" para configurar
2. **Upload**: Carregue seu arquivo CSV
3. **Análise**: Execute as análises disponíveis
4. **Insights**: Gere conclusões com IA
5. **Download**: Baixe relatórios completos

## ⚙️ Configuração

Esta aplicação requer uma API Key do Google Gemini configurada como variável de ambiente `GEMINI_API_KEY`.

Para obter sua chave:
1. Acesse: https://makersuite.google.com/app/apikey
2. Faça login com conta Google
3. Clique em "Create API Key"
4. Configure como secret no Hugging Face Space

## 🔗 Repositório

Código fonte disponível em: https://github.com/MarisaDeM/EDA_INTELIGENTE

## 📝 Limitações

- Suporte apenas para arquivos CSV
- Tamanho máximo recomendado: 50MB
- Algumas funcionalidades de exportação de imagem podem ter limitações

## 🤝 Contribuição

Contribuições são bem-vindas! Por favor, abra uma issue ou pull request no repositório GitHub.

## 📄 Licença

Apache License 2.0