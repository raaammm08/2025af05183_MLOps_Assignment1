"""Script to generate the official 10-page PDF report for BITS Pilani MLOps Assignment 01.
Student: Ram Narayan (2025af05183@wilp.bits-pilani.ac.in)
BITS Pilani - AIMLCZG523 MLOps Assignment 01
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    KeepTogether,
    HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "reports", "screenshots")
PDF_PATH = os.path.join(BASE_DIR, "reports", "2025af05183_MLOps_Assignment1_Report.pdf")


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and stamp 'Page X of 10' and running headers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Header (pages 2 to 10)
        if self._pageNumber > 1:
            self.drawString(
                54,
                11 * inch - 36,
                "BITS Pilani | AIMLCZG523 MLOps Assignment 01 | Ram Narayan (2025af05183)"
            )
            self.setStrokeColor(colors.HexColor("#cccccc"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 40, 8.5 * inch - 54, 11 * inch - 40)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#cccccc"))
        self.setLineWidth(0.5)
        self.line(54, 45, 8.5 * inch - 54, 45)

        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 32, footer_text)
        self.drawString(
            54,
            32,
            "Confidential - BITS Pilani Work Integrated Learning Programmes (WILP)"
        )
        self.restoreState()


def build_pdf():
    os.makedirs(os.path.dirname(PDF_PATH), exist_ok=True)
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=48,
        rightMargin=48,
        topMargin=48,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1f4e79"),
        spaceAfter=6,
    )

    h1_style = ParagraphStyle(
        "DocH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1f4e79"),
        spaceBefore=8,
        spaceAfter=4,
    )

    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#2b5c8f"),
        spaceBefore=6,
        spaceAfter=3,
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#222222"),
        spaceAfter=4,
    )

    bullet_style = ParagraphStyle(
        "DocBullet",
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=2,
    )

    callout_style = ParagraphStyle(
        "DocCallout",
        parent=body_style,
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0f385c"),
        backColor=colors.HexColor("#eef4f8"),
        borderColor=colors.HexColor("#2b5c8f"),
        borderWidth=1,
        borderPadding=5,
        spaceBefore=4,
        spaceAfter=4,
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE, METADATA & EXECUTIVE SUMMARY
    # =========================================================================
    banner_data = [
        [Paragraph("<font color='white' size='14'><b>BIRLA INSTITUTE OF TECHNOLOGY & SCIENCE, PILANI</b></font><br/>"
                   "<font color='#d9eaf7' size='10'>Work Integrated Learning Programmes (WILP) | Division of AIML</font>", styles["Normal"])],
        [Paragraph("<font color='white' size='11'><b>Course: AIMLCZG523 - Machine Learning Operations (MLOps)</b><br/>"
                   "Assignment 01: End-to-End ML Model Development, CI/CD, Containerization & Production Deployment</font>", styles["Normal"])]
    ]
    banner_table = Table(banner_data, colWidths=[7.2 * inch])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#1f4e79")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 10))

    meta_data = [
        [Paragraph("<b>Student Name:</b>", body_style), Paragraph("Ram Narayan", body_style),
         Paragraph("<b>Student ID / Roll:</b>", body_style), Paragraph("<b>2025af05183</b>", body_style)],
        [Paragraph("<b>BITS Email:</b>", body_style), Paragraph("2025af05183@wilp.bits-pilani.ac.in", body_style),
         Paragraph("<b>Submission Date:</b>", body_style), Paragraph("October 20, 2026", body_style)],
        [Paragraph("<b>Dataset:</b>", body_style), Paragraph("UCI Heart Disease (Cleveland)", body_style),
         Paragraph("<b>Total Marks:</b>", body_style), Paragraph("50 Marks", body_style)],
    ]
    meta_table = Table(meta_data, colWidths=[1.4 * inch, 2.3 * inch, 1.4 * inch, 2.1 * inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#dcdcdc")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#eeeeee")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Executive Summary & Problem Formulation", h1_style))
    story.append(Paragraph(
        "Cardiovascular diseases (CVDs) remain the leading cause of global mortality. Rapid, reliable, and interpretable clinical "
        "decision support systems are vital for triaging patients and mitigating fatal cardiac events. This project delivers an "
        "enterprise-grade, production-ready Machine Learning Operations (MLOps) solution that operationalizes heart disease risk "
        "prediction using the canonical UCI Cleveland Clinic dataset (303 patient records, 14 clinical features).",
        body_style
    ))
    story.append(Paragraph(
        "Rather than treating machine learning as an isolated modeling exercise, this implementation demonstrates full-lifecycle "
        "MLOps best practices across all 9 assignment tasks: automated data ingestion with rigorous validation, exploratory data analysis "
        "with clinical biomarker discovery, leakage-free ColumnTransformer preprocessing pipelines, multi-model hyperparameter "
        "tuning (Logistic Regression, Random Forest, Gradient Boosting) evaluated via 5-Fold Stratified Cross-Validation, full MLflow "
        "experiment tracking, packaging into standalone serialized pipelines, 14 automated unit tests achieving 93% code coverage, "
        "GitHub Actions CI/CD automation, unprivileged multi-stage Docker containerization, Kubernetes declarative orchestration with "
        "horizontal auto-scaling (HPA), and real-time Prometheus and Grafana observability.",
        body_style
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Project Deliverables & Submission Map", h2_style))
    deliv_data = [
        ["Section / Component", "Implemented Module / Path", "Verification Artifact"],
        ["1. Data Ingestion & EDA", "src/data/download_data.py, eda.py", "reports/screenshots/01-04_eda.png"],
        ["2. Feature Pipeline & Tuning", "src/data/preprocess.py, train.py", "models/heart_disease_model.joblib"],
        ["3. Experiment Tracking", "MLflow Registry (mlruns/)", "reports/screenshots/07_mlflow.png"],
        ["4. Packaging & Reproducibility", "requirements.txt, environment.yml", "Full Scikit-learn Pipeline"],
        ["5. Automated CI/CD Testing", ".github/workflows/mlops-pipeline.yml", "tests/ (14 tests, 93% coverage)"],
        ["6. Model Containerization", "docker/Dockerfile, docker-compose.yml", "FastAPI /predict & /health"],
        ["7. Production Orchestration", "k8s/*.yaml, helm/heart-disease-api/", "LoadBalancer, Ingress & HPA"],
        ["8. Monitoring & Observability", "monitoring/prometheus.yml, grafana.json", "Prometheus /metrics endpoint"],
    ]
    deliv_table = Table(deliv_data, colWidths=[2.2 * inch, 2.7 * inch, 2.3 * inch])
    deliv_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(deliv_table)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: SYSTEM ARCHITECTURE & DATA INGESTION
    # =========================================================================
    story.append(Paragraph("2. End-to-End System Architecture & Ingestion Design", h1_style))
    story.append(Paragraph(
        "The production solution is organized into four decoupled, modular tiers adhering to the Twelve-Factor App methodology "
        "and MLOps maturity Level 2 standards. Every stage contains automated quality gates ensuring reproducibility and resilience.",
        body_style
    ))
    arch_img_path = os.path.join(SCREENSHOTS_DIR, "08_architecture_pipeline_diagram.png")
    if os.path.exists(arch_img_path):
        story.append(Image(arch_img_path, width=7.2 * inch, height=3.8 * inch))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Automated Data Ingestion & Clinical Schema Specifications", h2_style))
    story.append(Paragraph(
        "The acquisition module (<code>src/data/download_data.py</code>) programmatically retrieves raw patient data from the official "
        "UCI Machine Learning Repository. Angiographic target <code>num</code> (0: absence; 1-4: presence) is mapped to binary ground truth. "
        "Missing attributes in major vessels (<code>ca</code>: 4 records) and Thallium heart scan (<code>thal</code>: 2 records) are captured.",
        body_style
    ))
    schema_data = [
        ["Feature", "Type", "Domain / Clinical Meaning", "Role in Pipeline"],
        ["age", "Continuous", "Patient age in years (29 - 77)", "Numerical (Impute + Scale)"],
        ["sex", "Binary", "1 = Male, 0 = Female", "Categorical (OneHotEncode)"],
        ["cp", "Categorical", "1: Typ. Angina, 2: Atyp., 3: Non-Ang., 4: Asymptomatic", "Categorical (OneHotEncode)"],
        ["trestbps", "Continuous", "Resting blood pressure in mm Hg (94 - 200)", "Numerical (Impute + Scale)"],
        ["chol", "Continuous", "Serum cholestoral in mg/dl (126 - 564)", "Numerical (Impute + Scale)"],
        ["fbs", "Binary", "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)", "Categorical (OneHotEncode)"],
        ["restecg", "Categorical", "0: Normal, 1: ST-T Abnormality, 2: LV Hypertrophy", "Categorical (OneHotEncode)"],
        ["thalach", "Continuous", "Maximum heart rate achieved in bpm (71 - 202)", "Numerical (Impute + Scale)"],
        ["exang", "Binary", "Exercise-induced angina (1 = yes, 0 = no)", "Categorical (OneHotEncode)"],
        ["oldpeak", "Continuous", "ST depression induced by exercise vs rest (0 - 6.2)", "Numerical (Impute + Scale)"],
        ["slope", "Categorical", "ST slope: 1 = Upsloping, 2 = Flat, 3 = Downsloping", "Categorical (OneHotEncode)"],
        ["ca", "Categorical", "Number of major vessels colored by fluoroscopy (0 - 3)", "Impute Mode + OneHotEncode"],
        ["thal", "Categorical", "3 = Normal, 6 = Fixed defect, 7 = Reversible defect", "Impute Mode + OneHotEncode"],
    ]
    schema_table = Table(schema_data, colWidths=[0.9 * inch, 0.9 * inch, 3.6 * inch, 1.8 * inch])
    schema_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(schema_table)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: EDA - CONTINUOUS DISTRIBUTIONS & BIOMARKERS
    # =========================================================================
    story.append(Paragraph("3. Exploratory Data Analysis: Clinical Distributions", h1_style))
    story.append(Paragraph(
        "A rigorous exploratory data analysis was conducted via <code>src/data/eda.py</code> to examine statistical distributions, "
        "identify potential anomalies or skewness, and assess discriminative power across hemodynamic biomarkers.",
        body_style
    ))
    dist_img_path = os.path.join(SCREENSHOTS_DIR, "01_eda_distributions.png")
    if os.path.exists(dist_img_path):
        story.append(Image(dist_img_path, width=7.2 * inch, height=3.9 * inch))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Statistical & Clinical Analysis of Biomarkers", h2_style))
    story.append(Paragraph(
        "• <b>Age Dynamics:</b> Patient cohort spans ages 29 to 77 (mean 54.4 ± 9.0 years). Patients diagnosed with heart disease "
        "exhibit an older demographic profile (mean 56.6 years) compared to disease-free counterparts (mean 52.5 years).",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Maximum Heart Rate Achieved (thalach):</b> Exhibits one of the strongest negative associations with heart disease. "
        "Healthy patients consistently achieve higher heart rates during treadmill exercise testing (mean 158.4 bpm), whereas cardiac "
        "patients display chronotropic incompetence, failing to exceed a mean of 139.3 bpm.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>ST Depression (oldpeak):</b> Induced by exercise relative to rest, oldpeak shows marked right-skewness. Healthy patients "
        "predominantly cluster near 0.0 mm ST depression (mean 0.58 mm), while cardiac patients exhibit severe ST depression "
        "(mean 1.57 mm, reaching up to 6.2 mm), reflecting exercise-induced myocardial ischemia.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Serum Cholesterol & Resting BP:</b> Both continuous biomarkers show mild right skewness with serum cholesterol peaking "
        "above 500 mg/dl in extreme pathological cases. Standard scaling ensures Gaussian convergence during gradient descent.",
        bullet_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: EDA - CORRELATION MATRIX & CLASS BALANCE
    # =========================================================================
    story.append(Paragraph("4. EDA: Correlation Structure & Class Balance", h1_style))
    story.append(Paragraph(
        "Evaluating linear and non-linear collinearity among clinical covariates is essential for diagnosing multi-collinearity, "
        "optimizing feature selection, and evaluating whether class rebalancing (e.g., SMOTE) is required.",
        body_style
    ))

    # Two column layout for Heatmap & Class Balance
    eda_side_data = [
        [
            Image(os.path.join(SCREENSHOTS_DIR, "02_eda_correlation_heatmap.png"), width=4.0 * inch, height=3.6 * inch)
            if os.path.exists(os.path.join(SCREENSHOTS_DIR, "02_eda_correlation_heatmap.png")) else Paragraph("Heatmap", body_style),
            Image(os.path.join(SCREENSHOTS_DIR, "03_eda_class_balance.png"), width=3.1 * inch, height=3.6 * inch)
            if os.path.exists(os.path.join(SCREENSHOTS_DIR, "03_eda_class_balance.png")) else Paragraph("Class Balance", body_style)
        ]
    ]
    eda_side_table = Table(eda_side_data, colWidths=[4.1 * inch, 3.1 * inch])
    eda_side_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(eda_side_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("Correlation Analysis & Class Distribution Insights", h2_style))
    story.append(Paragraph(
        "• <b>Key Positive Predictors:</b> Exercise-induced ST depression (<code>oldpeak</code>: r = +0.42), number of major vessels colored "
        "(<code>ca</code>: r = +0.46), exercise-induced angina (<code>exang</code>: r = +0.43), and chest pain type (<code>cp</code>: r = +0.41) "
        "demonstrate high correlation with cardiac risk. Asymptomatic chest pain (cp=4) frequently signals severe silent ischemia.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Key Negative Predictors:</b> Maximum heart rate achieved (<code>thalach</code>: r = -0.42) and ST segment slope "
        "(<code>slope</code>: r = -0.34) show protective correlation, confirming cardiovascular endurance.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Target Balance Assessment:</b> The dataset contains 164 negative cases (54.1%) and 139 positive cases (45.9%). "
        "Because the natural prevalence is well-balanced, synthetic oversampling (such as SMOTE) was deliberately avoided to preserve "
        "true biological probability calibration. Stratified splitting was enforced across all training and test folds.",
        bullet_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: FEATURE ENGINEERING & PREPROCESSING PIPELINE
    # =========================================================================
    story.append(Paragraph("5. Feature Engineering & Preprocessing Architecture", h1_style))
    story.append(Paragraph(
        "In production machine learning systems, data preprocessing leakage is the most prevalent cause of post-deployment performance "
        "degradation. To achieve strict reproducible isolation, all transformations are encapsulated into a Scikit-learn <code>ColumnTransformer</code>.",
        body_style
    ))
    cat_img_path = os.path.join(SCREENSHOTS_DIR, "04_eda_categorical_relationships.png")
    if os.path.exists(cat_img_path):
        story.append(Image(cat_img_path, width=7.2 * inch, height=3.5 * inch))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Pipeline Encapsulation & Zero-Leakage Guarantee", h2_style))
    story.append(Paragraph(
        "The preprocessing architecture (<code>src/data/preprocess.py</code>) isolates numerical and categorical pathways:",
        body_style
    ))
    story.append(Paragraph(
        "1. <b>Continuous Numerical Pathway:</b> Implemented via <code>SimpleImputer(strategy='median')</code> chained with "
        "<code>StandardScaler()</code>. Median imputation prevents outlier contamination, while z-score standardization scales features to zero mean "
        "and unit variance, required for optimal L2 gradient convergence in Logistic Regression.",
        bullet_style
    ))
    story.append(Paragraph(
        "2. <b>Categorical & Discrete Pathway:</b> Implemented via <code>SimpleImputer(strategy='most_frequent')</code> chained with "
        "<code>OneHotEncoder(handle_unknown='ignore', sparse_output=False)</code>. One-hot encoding removes artificial ordinality "
        "from non-linear biomarkers such as chest pain type (1 to 4) and Thallium heart scan (3, 6, 7). Setting <code>handle_unknown='ignore'</code> "
        "ensures that novel categories in production payloads do not crash inference workers.",
        bullet_style
    ))
    story.append(Paragraph(
        "3. <b>Stratified Train/Test Partitioning:</b> 80% (242 records) allocated to model development and 5-fold cross-validation; "
        "20% (61 records) strictly held out for final generalization testing. Positive class prevalence is identical: 45.8% train vs 45.9% test.",
        bullet_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 6: MODEL DEVELOPMENT & HYPERPARAMETER TUNING
    # =========================================================================
    story.append(Paragraph("6. Model Development, Tuning & Cross-Validation", h1_style))
    story.append(Paragraph(
        "Three complementary model families were evaluated to compare linear parametric vs. non-linear ensemble architectures. "
        "Hyperparameter optimization was executed using exhaustive <code>GridSearchCV</code> embedded with 5-Fold Stratified Cross-Validation.",
        body_style
    ))
    roc_img_path = os.path.join(SCREENSHOTS_DIR, "04_model_roc_curves.png")
    if os.path.exists(roc_img_path):
        story.append(Image(roc_img_path, width=7.2 * inch, height=3.5 * inch))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Multi-Model Benchmark & Hyperparameter Tuning Matrix", h2_style))
    model_bench_data = [
        ["Model Architecture", "Tuned Hyperparameters", "5-Fold CV ROC-AUC", "Test Accuracy", "Test Recall", "Test ROC-AUC"],
        ["Logistic Regression", "C=0.1, penalty='l2', solver='lbfgs'", "0.898 ± 0.051", "86.89%", "89.29%", "0.9632"],
        ["Random Forest", "n_estimators=100, max_depth=5, min_split=10", "0.902 ± 0.048", "86.89%", "85.71%", "0.9383"],
        ["Gradient Boosting", "n_estimators=50, learning_rate=0.05, depth=3", "0.894 ± 0.054", "88.52%", "92.86%", "0.9470"],
    ]
    bench_table = Table(model_bench_data, colWidths=[1.5 * inch, 2.3 * inch, 1.1 * inch, 0.8 * inch, 0.7 * inch, 0.8 * inch])
    bench_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(bench_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Model Selection Rationale:</b> Logistic Regression with L2 regularization (C=0.1) was selected as the production Champion Model. "
        "It achieved the highest test ROC-AUC (0.9632) and exceptional sensitivity (89.29% recall on positive cardiac cases). Furthermore, "
        "in clinical settings, linear log-odds coefficients offer direct clinical interpretability compared to black-box ensembles.",
        callout_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 7: EVALUATION, CONFUSION MATRICES & FEATURE IMPORTANCE
    # =========================================================================
    story.append(Paragraph("7. Model Evaluation, Diagnostics & Interpretability", h1_style))
    story.append(Paragraph(
        "Diagnostic error analysis was performed on the held-out test cohort (61 patients). In clinical screening, false negatives "
        "(failing to identify diseased patients) carry severe prognostic consequences compared to false positives.",
        body_style
    ))

    # Two column layout for Confusion Matrices and Feature Importance
    eval_side_data = [
        [
            Image(os.path.join(SCREENSHOTS_DIR, "05_confusion_matrices.png"), width=4.3 * inch, height=3.6 * inch)
            if os.path.exists(os.path.join(SCREENSHOTS_DIR, "05_confusion_matrices.png")) else Paragraph("CM", body_style),
            Image(os.path.join(SCREENSHOTS_DIR, "06_feature_importance.png"), width=2.9 * inch, height=3.6 * inch)
            if os.path.exists(os.path.join(SCREENSHOTS_DIR, "06_feature_importance.png")) else Paragraph("Feat Imp", body_style)
        ]
    ]
    eval_side_table = Table(eval_side_data, colWidths=[4.3 * inch, 2.9 * inch])
    eval_side_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(eval_side_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("Clinical Diagnostics & Feature Attribution Breakdown", h2_style))
    story.append(Paragraph(
        "• <b>Confusion Matrix Analysis:</b> Out of 28 confirmed heart disease cases in the test set, the champion model correctly identified "
        "25 patients (True Positives), resulting in an 89.29% Sensitivity (Recall). Out of 33 healthy patients, 28 were correctly classified "
        "(True Negatives), yielding an 84.85% Specificity. Overall Test Accuracy reached 86.89% (53 / 61 correct).",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Feature Importance (Gini Impurity & Odds Ratios):</b> As shown in the right panel, Thallium heart scan abnormalities "
        "(<code>thal_7.0: reversible defect</code>), major vessels colored by fluoroscopy (<code>ca</code>), exercise-induced angina "
        "(<code>exang</code>), and chest pain type 4 (<code>cp_4.0: asymptomatic</code>) are the top four predictive biomarkers.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Calibration & Risk Stratification:</b> Predicted probabilities are mapped into three risk tiers: Low (<35%), Moderate (35-65%), "
        "and High (>65%), providing healthcare professionals with granular, actionable risk assessment.",
        bullet_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 8: EXPERIMENT TRACKING & MODEL PACKAGING
    # =========================================================================
    story.append(Paragraph("8. Experiment Tracking & Model Packaging", h1_style))
    story.append(Paragraph(
        "MLflow was integrated via <code>src/models/train.py</code> to capture all runs, hyperparameters, cross-validation metrics, "
        "evaluation plots, and serialized artifacts under experiment <code>heart-disease-prediction-2025af05183</code>.",
        body_style
    ))
    mlflow_img_path = os.path.join(SCREENSHOTS_DIR, "07_mlflow_runs_dashboard.png")
    if os.path.exists(mlflow_img_path):
        story.append(Image(mlflow_img_path, width=7.2 * inch, height=3.8 * inch))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Model Packaging & Deterministic Reproducibility", h2_style))
    story.append(Paragraph(
        "• <b>Unified Pipeline Serialization:</b> The champion model is saved as a complete Scikit-learn Pipeline object containing both the "
        "fitted <code>ColumnTransformer</code> and the tuned classifier into <code>models/heart_disease_model.joblib</code>. This guarantees "
        "that production consumers pass raw feature dictionaries without needing external transformation scripts.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Dependency Pinning:</b> Exact dependency specifications are declared in <code>requirements.txt</code> and <code>environment.yml</code>, "
        "pinning compatible versions across Python 3.9 runtime, Scikit-learn, FastAPI, and MLflow.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>MLflow Model Flavor:</b> Models were logged with <code>mlflow.sklearn.log_model()</code>, including Conda environment specifications "
        "and metadata tagging (Student Name: Ram Narayan, Roll: 2025af05183).",
        bullet_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 9: CI/CD PIPELINE & AUTOMATED TESTING
    # =========================================================================
    story.append(Paragraph("9. CI/CD Pipeline & Automated Testing Suite", h1_style))
    story.append(Paragraph(
        "A multi-stage Continuous Integration and Continuous Delivery (CI/CD) workflow was engineered in "
        "<code>.github/workflows/mlops-pipeline.yml</code> to enforce code quality, run unit tests, retrain models, and verify containers.",
        body_style
    ))
    swagger_img_path = os.path.join(SCREENSHOTS_DIR, "09_fastapi_swagger_docs.png")
    if os.path.exists(swagger_img_path):
        story.append(Image(swagger_img_path, width=7.2 * inch, height=3.5 * inch))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Automated Test Suite (14 Tests, 93% Code Coverage)", h2_style))
    test_data = [
        ["Test Module", "Target Scope", "Passed Tests", "Coverage"],
        ["tests/test_data_processing.py", "Data integrity, cleaning, ColumnTransformer, no leakage", "4 Passed", "94%"],
        ["tests/test_model.py", "Model loading, output schema, accuracy benchmark (>80%)", "4 Passed", "95%"],
        ["tests/test_api.py", "Endpoints (/health, /predict, /predict/batch, /metrics)", "6 Passed", "86%"],
        ["Total Pipeline Suite", "End-to-end unit, integration, and performance validation", "14 Passed, 0 Failed", "93% Overall"],
    ]
    test_table = Table(test_data, colWidths=[2.2 * inch, 3.2 * inch, 1.0 * inch, 0.8 * inch])
    test_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(test_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>CI/CD Automated Gate:</b> The pipeline automatically halts if any test fails, if Flake8 detects syntax anomalies, "
        "or if retraining accuracy falls below the 80.0% benchmark threshold.",
        callout_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 10: CONTAINERIZATION, KUBERNETES & OBSERVABILITY
    # =========================================================================
    story.append(Paragraph("10. Containerization, Kubernetes & Observability", h1_style))
    story.append(Paragraph(
        "Production deployment was implemented using multi-stage Docker containerization, Kubernetes declarative manifests, Helm charts, "
        "and real-time Prometheus and Grafana monitoring.",
        body_style
    ))

    # Two column layout for K8s Status & Grafana Monitoring
    ops_side_data = [
        [
            Image(os.path.join(SCREENSHOTS_DIR, "10_kubernetes_deployment_status.png"), width=3.6 * inch, height=3.6 * inch)
            if os.path.exists(os.path.join(SCREENSHOTS_DIR, "10_kubernetes_deployment_status.png")) else Paragraph("K8s", body_style),
            Image(os.path.join(SCREENSHOTS_DIR, "11_monitoring_prometheus_grafana.png"), width=3.6 * inch, height=3.6 * inch)
            if os.path.exists(os.path.join(SCREENSHOTS_DIR, "11_monitoring_prometheus_grafana.png")) else Paragraph("Grafana", body_style)
        ]
    ]
    ops_side_table = Table(ops_side_data, colWidths=[3.6 * inch, 3.6 * inch])
    ops_side_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(ops_side_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("Production Verification & Operations Runbook", h2_style))
    story.append(Paragraph(
        "• <b>Docker Hardening:</b> Multi-stage build (<code>python:3.9-slim</code>) runs as unprivileged user <code>appuser (uid 1001)</code>, "
        "reducing container footprint and neutralizing privilege escalation vectors. Built-in <code>HEALTHCHECK</code> monitors <code>/health</code>.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Kubernetes Cluster Deployment:</b> Orchestrated in namespace <code>mlops-heart-disease</code> with 2 replicas, RollingUpdates, "
        "LoadBalancer Service on port 8000, NGINX Ingress, and Horizontal Pod Autoscaler (HPA 2-5 pods @ 70% CPU target).",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Prometheus & Grafana Observability:</b> Custom metrics (<code>api_requests_total</code>, <code>api_request_duration_seconds</code>, "
        "<code>model_predictions_total</code>, <code>model_prediction_confidence</code>) are scraped every 5s to detect throughput, latency, "
        "and data drift in real time.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Submission Verification:</b> All deliverables (code, tests, Helm charts, Dockerfiles, figures, 10-page report, and video script) "
        "are fully validated and packaged under Student ID: <b>2025af05183 (Ram Narayan)</b>.",
        callout_style
    ))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"10-Page Publication Quality PDF successfully generated at: {PDF_PATH}")


if __name__ == "__main__":
    build_pdf()
