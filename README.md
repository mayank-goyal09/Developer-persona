# Developer Persona Segmentation

Built a developer persona segmentation project using the **Stack Overflow 2025 Annual Developer Survey dataset** (public responses + schema), where survey data was cleaned for high missingness, multi-select tech-stack answers were engineered into token features (e.g., languages/platforms/databases), and a **scikit-learn preprocessing pipeline** (imputation + one-hot encoding + scaling) was used to train a **MiniBatch K-Means clustering model** that grouped **~42K respondents** into three interpretable personas—such as a large **"Modern Web Builders"** cluster driven by JavaScript-related stack signals, a mid-sized generalist cluster with broader platform/database selections, and a smaller **"Veteran Builders"** cluster with very high WorkExp/YearsCode—producing a persona report and labeled dataset for use in hiring, developer marketing, and community/product strategy.

## 🎯 Project Overview

This project segments developers into distinct personas based on their survey responses, enabling data-driven insights for:
- **Hiring strategies** - Target specific developer archetypes
- **Developer marketing** - Tailor messaging to different segments
- **Community & product strategy** - Build features aligned with user needs

## 📊 Dataset

- **Source**: Stack Overflow 2025 Annual Developer Survey
- **Size**: ~42,000 respondents
- **Features**: Multi-select tech stack (languages, platforms, databases), work experience, years coding, and more

## 🔧 Technical Approach

### Data Preprocessing
1. **Cleaning**: Handled high missingness in survey responses
2. **Feature Engineering**: 
   - Tokenized multi-select tech-stack answers
   - Created binary features for languages, platforms, and databases
3. **Pipeline**: 
   - Imputation for missing values
   - One-hot encoding for categorical variables
   - Standard scaling for numerical features

### Clustering Model
- **Algorithm**: MiniBatch K-Means
- **Number of Clusters**: 3 interpretable personas
- **Library**: scikit-learn

## 👥 Developer Personas Identified

### 1. Modern Web Builders (Largest Cluster)
- **Characteristics**: JavaScript-heavy tech stack
- **Key Technologies**: React, Node.js, TypeScript, modern web frameworks
- **Profile**: Frontend-focused, web-centric developers

### 2. Generalist Developers (Mid-sized Cluster)
- **Characteristics**: Diverse platform and database selections
- **Key Technologies**: Multiple languages, varied databases and platforms
- **Profile**: Full-stack versatility, adaptable skill sets

### 3. Veteran Builders (Smaller Cluster)
- **Characteristics**: High WorkExp and YearsCode values
- **Key Technologies**: Established languages, mature platforms
- **Profile**: Senior developers with extensive experience

## 📁 Repository Contents

- `app.py` - Streamlit dashboard for persona visualization
- `cluster_persona_report.csv` - Detailed persona analysis report
- `stack_overflow_2025_segmented_users.csv` - Labeled dataset with cluster assignments
- `survey_results_schema.csv` - Survey schema documentation

## 🚀 Usage

Run the Streamlit app to explore the personas:
```bash
streamlit run app.py
```

## 💡 Applications

- **Recruitment**: Target job postings to specific personas
- **Product Development**: Build features aligned with persona needs
- **Marketing Campaigns**: Craft messaging for each segment
- **Community Building**: Create content and events tailored to different developer types

## 🛠️ Tech Stack

- **Python** - Core programming language
- **scikit-learn** - Machine learning and preprocessing
- **pandas** - Data manipulation
- **Streamlit** - Interactive visualization dashboard

---

*Built as part of my ML portfolio showcasing unsupervised learning and real-world data analysis.*