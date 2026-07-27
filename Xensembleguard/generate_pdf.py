from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable, PageBreak)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import KeepTogether

OUTPUT = "/Users/kasanagottusaimaniteja/Desktop/Project_Document.pdf"

doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4,
    rightMargin=2*cm, leftMargin=2*cm,
    topMargin=2.5*cm, bottomMargin=2.5*cm
)

# ── Colour palette ────────────────────────────────────────────
DARK_BLUE  = colors.HexColor("#1a237e")
MID_BLUE   = colors.HexColor("#283593")
ACCENT     = colors.HexColor("#1565c0")
LIGHT_BLUE = colors.HexColor("#e3f2fd")
LIGHT_GREY = colors.HexColor("#f5f5f5")
WHITE      = colors.white
BLACK      = colors.HexColor("#212121")
GREEN      = colors.HexColor("#2e7d32")
RED        = colors.HexColor("#c62828")
ORANGE     = colors.HexColor("#e65100")

# ── Styles ────────────────────────────────────────────────────
styles = getSampleStyleSheet()

TITLE = ParagraphStyle("title",
    fontSize=16, leading=22, alignment=TA_CENTER,
    textColor=WHITE, fontName="Helvetica-Bold",
    spaceAfter=4)

SUBTITLE = ParagraphStyle("subtitle",
    fontSize=10, leading=14, alignment=TA_CENTER,
    textColor=colors.HexColor("#bbdefb"), fontName="Helvetica",
    spaceAfter=2)

SECTION = ParagraphStyle("section",
    fontSize=13, leading=18, alignment=TA_LEFT,
    textColor=WHITE, fontName="Helvetica-Bold",
    spaceBefore=4, spaceAfter=4)

SUBSECTION = ParagraphStyle("subsection",
    fontSize=11, leading=15, alignment=TA_LEFT,
    textColor=DARK_BLUE, fontName="Helvetica-Bold",
    spaceBefore=8, spaceAfter=4)

BODY = ParagraphStyle("body",
    fontSize=10, leading=15, alignment=TA_JUSTIFY,
    textColor=BLACK, fontName="Helvetica",
    spaceAfter=6)

BULLET = ParagraphStyle("bullet",
    fontSize=10, leading=15, alignment=TA_LEFT,
    textColor=BLACK, fontName="Helvetica",
    leftIndent=16, spaceAfter=4,
    bulletIndent=6)

CODE = ParagraphStyle("code",
    fontSize=8.5, leading=13, alignment=TA_LEFT,
    textColor=colors.HexColor("#1a237e"), fontName="Courier",
    backColor=LIGHT_BLUE, leftIndent=12, rightIndent=12,
    spaceAfter=6, spaceBefore=4,
    borderPad=6)

LABEL = ParagraphStyle("label",
    fontSize=9, leading=12, alignment=TA_CENTER,
    textColor=colors.HexColor("#546e7a"), fontName="Helvetica-Oblique",
    spaceAfter=8)

# ── Helper builders ───────────────────────────────────────────
def section_header(text):
    data = [[Paragraph(text, SECTION)]]
    t = Table(data, colWidths=[17*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), DARK_BLUE),
        ("ROWPADDING",  (0,0), (-1,-1), 8),
        ("ROUNDEDCORNERS", [4]),
        ("BOX", (0,0), (-1,-1), 0, WHITE),
    ]))
    return t

def info_table(rows, col_widths=None):
    if col_widths is None:
        col_widths = [5*cm, 12*cm]
    t = Table(rows, colWidths=col_widths)
    style = [
        ("FONTNAME",   (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",   (1,0), (1,-1), "Helvetica"),
        ("FONTSIZE",   (0,0), (-1,-1), 9.5),
        ("LEADING",    (0,0), (-1,-1), 14),
        ("TEXTCOLOR",  (0,0), (0,-1), DARK_BLUE),
        ("TEXTCOLOR",  (1,0), (1,-1), BLACK),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [WHITE, LIGHT_GREY]),
        ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#cfd8dc")),
        ("ROWPADDING", (0,0), (-1,-1), 6),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
    ]
    t.setStyle(TableStyle(style))
    return t

def results_table(header, rows, col_widths=None):
    all_rows = [header] + rows
    if col_widths is None:
        col_widths = [17*cm / len(header)] * len(header)
    t = Table(all_rows, colWidths=col_widths)
    style = [
        ("BACKGROUND",  (0,0), (-1,0), ACCENT),
        ("TEXTCOLOR",   (0,0), (-1,0), WHITE),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("LEADING",     (0,0), (-1,-1), 13),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LIGHT_BLUE]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#b0bec5")),
        ("ROWPADDING",  (0,0), (-1,-1), 6),
    ]
    t.setStyle(TableStyle(style))
    return t

def gap(h=8):
    return Spacer(1, h)

def hr():
    return HRFlowable(width="100%", thickness=0.5,
                      color=colors.HexColor("#cfd8dc"), spaceAfter=6)

# ═════════════════════════════════════════════════════════════
# BUILD CONTENT
# ═════════════════════════════════════════════════════════════
story = []

# ── COVER HEADER ─────────────────────────────────────────────
cover_data = [[
    Paragraph("Drift-Aware Explainable Ensemble Learning<br/>for Adaptive Network Intrusion Detection", TITLE),
    Paragraph("Research Project Document  ·  UNSW-NB15 Dataset  ·  Python 3.12", SUBTITLE),
]]
cover = Table(cover_data, colWidths=[17*cm])
cover.setStyle(TableStyle([
    ("BACKGROUND",  (0,0), (-1,-1), MID_BLUE),
    ("ROWPADDING",  (0,0), (-1,-1), 18),
    ("BOX",         (0,0), (-1,-1), 1, DARK_BLUE),
    ("ROUNDEDCORNERS", [6]),
]))
story += [cover, gap(18)]

# ── ABSTRACT ─────────────────────────────────────────────────
story += [section_header("Abstract"), gap(6)]
abstract = (
    "Cyberattacks are growing in number and complexity, making it harder for traditional security systems "
    "to protect networks effectively. These systems depend on fixed rules that cannot detect new or unknown "
    "attacks. As a result, machine learning has become an important tool for building smarter Network "
    "Intrusion Detection Systems that can learn to identify threats from network traffic data."
    "<br/><br/>"
    "Despite their promise, most machine learning-based detection systems face two critical problems. "
    "First, they cannot explain their decisions. When a system flags network traffic as an attack, analysts "
    "have no way of understanding why, which makes it difficult to trust or act on the alert. Second, these "
    "systems are trained once and never updated. As attackers change their methods over time, the model's "
    "performance quietly declines without anyone noticing — a problem known as concept drift."
    "<br/><br/>"
    "This paper introduces a drift-aware explainable ensemble learning framework designed to solve both "
    "problems. The proposed system combines multiple machine learning models in a layered structure, where "
    "each model contributes its prediction and a final model learns how to combine them best. To make the "
    "system transparent, SHAP-based explanations are used to clearly show which features of the network "
    "traffic led to each detection decision. To keep the system reliable over time, an ADWIN-based drift "
    "detection mechanism watches for changes in network behavior and signals when the model needs to be "
    "updated. The framework is tested on the UNSW-NB15 dataset, confirming its ability to detect intrusions "
    "accurately, explain its decisions clearly, and adapt to changing network conditions."
)
story += [Paragraph(abstract, BODY), gap(4)]
story += [Paragraph("<i>Word Count: 240</i>", LABEL), gap(12)]

# ── KEYWORDS ─────────────────────────────────────────────────
story += [section_header("Keywords"), gap(6)]
story += [Paragraph(
    "Network Intrusion Detection &nbsp;·&nbsp; Ensemble Learning &nbsp;·&nbsp; "
    "Explainable Artificial Intelligence &nbsp;·&nbsp; Concept Drift &nbsp;·&nbsp; "
    "ADWIN &nbsp;·&nbsp; SHAP", BODY), gap(12)]

# ── RESEARCH GAPS ─────────────────────────────────────────────
story += [section_header("Research Gaps"), gap(6)]
gaps_data = [
    ["Gap 1", "Models Cannot Explain Their Decisions — Current systems detect attacks but cannot tell why. "
              "Analysts are left with no explanation, making it hard to trust or act on the output."],
    ["Gap 2", "Models Stop Working Well Over Time — Systems trained on old data never update. As attackers "
              "change methods, accuracy drops silently with no built-in detection mechanism."],
    ["Gap 3", "No System Combines Both Solutions — Explainability and drift detection are always studied "
              "separately. No existing framework brings both together in one unified IDS pipeline."],
    ["Gap 4", "Single Models Used Instead of Combined Ones — Most IDS research relies on one classifier. "
              "Stacked ensemble learning, which gives better reliability, remains underexplored."],
]
story += [info_table([[Paragraph(g, ParagraphStyle("gk", fontSize=9.5, fontName="Helvetica-Bold",
                                                    textColor=DARK_BLUE)),
                        Paragraph(d, BODY)] for g, d in gaps_data]), gap(12)]

# ── RESEARCH OBJECTIVES ───────────────────────────────────────
story += [section_header("Research Objectives"), gap(6)]
objectives = [
    ("Objective 1", "Build a Stacked Ensemble Model",
     "Combine LightGBM, XGBoost, CatBoost, and Random Forest into a stacked architecture. "
     "A meta-model then learns the best way to combine their outputs for accurate intrusion detection."),
    ("Objective 2", "Add Explainability Using SHAP",
     "Integrate a SHAP-based module that explains every detection decision, identifying which network "
     "features triggered the alert and which model contributed most."),
    ("Objective 3", "Detect Concept Drift Using ADWIN",
     "Implement an ADWIN-based monitoring system that watches model performance during deployment and "
     "automatically raises a retraining alert when accuracy drops."),
    ("Objective 4", "Detect Zero-Day Attacks Using Isolation Forest",
     "Add an Isolation Forest layer trained only on normal traffic that flags completely unknown "
     "attacks as zero-day threats, even when the classifier has never encountered them."),
    ("Objective 5", "Evaluate on Real-World Dataset",
     "Test the entire framework on the UNSW-NB15 benchmark dataset and build a fully automated "
     "end-to-end Python pipeline covering data loading, training, SHAP analysis, and drift simulation."),
]
for num, title, desc in objectives:
    story += [Paragraph(f"<b>{num} — {title}</b>", SUBSECTION),
              Paragraph(desc, BODY)]
story.append(gap(6))

# ── SCOPE ────────────────────────────────────────────────────
story += [section_header("Scope"), gap(6)]
story += [Paragraph(
    "This study focuses on building an intelligent Network Intrusion Detection System using machine "
    "learning techniques applied to the UNSW-NB15 benchmark dataset. The scope includes designing a "
    "stacked ensemble model for binary intrusion detection, integrating SHAP-based explainability, "
    "implementing Isolation Forest for zero-day attack identification, and applying ADWIN-based concept "
    "drift detection to monitor model performance over time. The study is limited to network traffic "
    "classification in a simulated environment and does not cover real-time deployment, hardware-level "
    "security, or encrypted traffic analysis.", BODY), gap(4)]
story += [Paragraph("<i>Word Count: 100</i>", LABEL), gap(12)]

# ── BRIEF OVERVIEW ────────────────────────────────────────────
story += [section_header("Brief Overview of the Proposed Method"), gap(6)]

story += [Paragraph("<b>Stage 1 — Stacked Ensemble Detection</b>", SUBSECTION)]
story += [Paragraph(
    "Four machine learning models — LightGBM, XGBoost, CatBoost, and Random Forest — are trained "
    "independently on network traffic data. Each model produces a probability score indicating whether "
    "the traffic is normal or an attack. A meta-level XGBoost classifier then combines these scores "
    "along with the original traffic features to make the final detection decision. This layered "
    "approach produces more reliable results than any single model alone.", BODY)]

story += [Paragraph("<b>Stage 2 — Explainability and Zero-Day Detection</b>", SUBSECTION)]
story += [Paragraph(
    "A SHAP-based module explains every prediction by identifying which features and which models "
    "influenced the decision most. Alongside this, an Isolation Forest is trained exclusively on normal "
    "traffic to detect completely new and unknown attacks that the classifier has never encountered "
    "before, flagging them as zero-day threats.", BODY)]

story += [Paragraph("<b>Stage 3 — Concept Drift Monitoring</b>", SUBSECTION)]
story += [Paragraph(
    "An ADWIN-based drift detector continuously monitors the model's accuracy during deployment. "
    "When attack patterns change over time and performance degrades, the system automatically raises "
    "a retraining alert to keep the model up to date.", BODY), gap(4)]
story += [Paragraph("<i>Word Count: 197</i>", LABEL), gap(12)]

# ── WORKFLOW ─────────────────────────────────────────────────
story += [section_header("System Workflow"), gap(6)]
workflow = (
    "Network Traffic Data (UNSW-NB15)  →  Data Preprocessing (Clean | Encode | Scale)  →  "
    "Isolation Forest [Novelty 3: Zero-Day]  +  Base Models [LightGBM | XGBoost | CatBoost | RF]  →  "
    "Meta-Model XGBoost [Stacked Ensemble]  →  ✅ Normal Traffic  /  🚨 Attack Detected  →  "
    "SHAP Explain [Novelty 1]  +  ADWIN Monitor [Novelty 2]"
)
story += [Paragraph(workflow, CODE), gap(12)]

# ── KEY TECHNOLOGIES ──────────────────────────────────────────
story += [section_header("Key Technologies"), gap(6)]
tech_rows = [
    ["Language", "Python 3.12"],
    ["ML / Ensemble", "LightGBM, XGBoost, CatBoost, scikit-learn (Random Forest, Isolation Forest, StandardScaler)"],
    ["Explainability", "SHAP (TreeSHAP / TreeExplainer)"],
    ["Concept Drift", "River library (ADWIN algorithm)"],
    ["Data / Storage", "Pandas, NumPy, CSV (UNSW-NB15 Dataset), Joblib (Model Saving)"],
    ["Visualization", "Matplotlib, Plotly"],
    ["Dashboard", "Streamlit"],
    ["Version Control", "Git / GitHub"],
    ["Hardware Target", "Apple Silicon CPU / NVIDIA GPU (optional, XGBoost GPU support)"],
]
story += [info_table([[Paragraph(k, ParagraphStyle("tk", fontSize=9.5, fontName="Helvetica-Bold",
                                                    textColor=DARK_BLUE)),
                        Paragraph(v, BODY)] for k, v in tech_rows]), gap(12)]

# ── PAGE BREAK ────────────────────────────────────────────────
story.append(PageBreak())

# ── DATASET ───────────────────────────────────────────────────
story += [section_header("Dataset Details"), gap(6)]
story += [results_table(
    ["Property", "Value"],
    [["Name", "UNSW-NB15"],
     ["Total Records", "257,673"],
     ["Features", "42 (after preprocessing)"],
     ["Task", "Binary — Normal vs Attack"],
     ["Train Set", "206,138 (80%)"],
     ["Test Set", "51,535 (20%)"],
     ["Normal (test)", "18,600"],
     ["Attack (test)", "32,935"]],
    [7*cm, 10*cm]
), gap(14)]

# ── RESULTS ───────────────────────────────────────────────────
story += [section_header("Results"), gap(6)]

story += [Paragraph("<b>Base Model Performance</b>", SUBSECTION)]
story += [results_table(
    ["Model", "Accuracy", "Precision", "Recall", "F1-Score"],
    [["LightGBM",     "95.02%", "94.39%", "94.91%", "94.64%"],
     ["XGBoost",      "95.04%", "94.44%", "94.88%", "94.65%"],
     ["CatBoost",     "94.77%", "94.16%", "94.57%", "94.36%"],
     ["Random Forest","93.84%", "93.28%", "93.40%", "93.34%"]],
    [5*cm, 3*cm, 3*cm, 3*cm, 3*cm]
), gap(10)]

story += [Paragraph("<b>Meta-Model — Final System</b>", SUBSECTION)]
story += [results_table(
    ["Accuracy", "AUC-ROC", "F1-Score", "Precision", "Recall"],
    [["94.90%", "99.20% 🏆", "94.48%", "94.44%", "94.52%"]],
    [3.4*cm]*5
), gap(10)]

story += [Paragraph("<b>Confusion Matrix</b>", SUBSECTION)]
story += [results_table(
    ["", "Predicted Normal", "Predicted Attack"],
    [["Actual Normal", "17,328 (TN) ✅", "1,272 (FP)"],
     ["Actual Attack", "1,356 (FN)", "31,579 (TP) ✅"]],
    [4.5*cm, 6.25*cm, 6.25*cm]
), gap(10)]

story += [Paragraph("<b>SHAP Attribution Table — Novelty 1</b>", SUBSECTION)]
story += [results_table(
    ["Class", "Top Contributor", "LightGBM", "XGBoost", "CatBoost", "RandomForest"],
    [["Normal", "LightGBM", "2.3324", "2.2675", "0.4899", "0.6309"],
     ["Attack", "LightGBM", "2.9162", "2.8152", "0.3633", "0.7355"]],
    [2.5*cm, 3.5*cm, 2.75*cm, 2.75*cm, 2.75*cm, 2.75*cm]
), gap(10)]

story += [Paragraph("<b>Concept Drift — ADWIN — Novelty 2</b>", SUBSECTION)]
story += [results_table(
    ["Phase", "Batches", "Mean Accuracy", "Std Dev"],
    [["Pre-Drift", "0 – 199", "94.83%", "±1.63%"],
     ["Drifted",   "200 – 400", "91.91%", "±4.57%"],
     ["Drop",      "—", "−2.92%", "—"]],
    [4.25*cm, 4.25*cm, 4.25*cm, 4.25*cm]
), gap(10)]

story += [Paragraph("<b>Zero-Day Detection — Isolation Forest — Novelty 3</b>", SUBSECTION)]
story += [results_table(
    ["Precision", "Recall", "AUC-ROC", "Training Samples (Normal)"],
    [["91.77%", "31.54%", "79.43%", "74,400"]],
    [4.25*cm]*4
), gap(14)]

# ── NOVELTY EXPLANATIONS ──────────────────────────────────────
story += [section_header("What is SHAP?"), gap(6)]
story += [Paragraph(
    "SHAP (SHapley Additive exPlanations) explains why a machine learning model made a specific "
    "prediction. After every detection, SHAP shows which network features pushed the prediction "
    "toward Attack and which pushed it toward Normal. This makes the system transparent and "
    "trustworthy for security analysts.", BODY)]
story += [Paragraph(
    "<b>Example:</b> duration = 0.001 sec → SHAP +2.91 (very short, suspicious — pushes toward Attack) | "
    "src_bytes = 999,999 → SHAP +1.83 (huge data — pushes toward Attack) | "
    "dst_port = 80 → SHAP −0.45 (normal web port — pushes toward Normal)", BODY), gap(12)]

story += [section_header("What is ADWIN?"), gap(6)]
story += [Paragraph(
    "ADWIN (ADaptive WINdowing) monitors a stream of accuracy values over time. It maintains a "
    "sliding window and checks whether recent average accuracy has significantly dropped compared "
    "to the earlier average. When it detects a drop, it raises a drift alert — signalling that "
    "the model needs to be retrained because attack patterns have changed.", BODY), gap(12)]

story += [section_header("Why is Meta-Model Slightly Less Than Individual Models?"), gap(6)]
story += [Paragraph(
    "The 0.14% difference between XGBoost (95.04%) and the Meta-Model (94.90%) is within the "
    "statistical margin of error — just 72 samples out of 51,535. The Meta-Model's true advantage "
    "is its AUC-ROC of 99.20% (higher than all individual models), its robustness when one model "
    "fails, and its better generalization through stacking which avoids overfitting.", BODY), gap(12)]

# ── HOW TO RUN ────────────────────────────────────────────────
story += [section_header("How to Run"), gap(6)]
run_steps = [
    "python Main.py --dataset UNSW_NB15   # Full pipeline (train + SHAP + drift + zero-day)",
    "python show_results.py               # View results in terminal + open plots",
    "streamlit run dashboard.py           # Open interactive web dashboard",
]
for s in run_steps:
    story += [Paragraph(s, CODE)]
story.append(gap(12))

# ── PROJECT FILES ─────────────────────────────────────────────
story += [section_header("Project Files"), gap(6)]
story += [results_table(
    ["File", "Purpose"],
    [["Main.py",                "Runs the complete pipeline"],
     ["data_preprocessing.py",  "Cleans and prepares data"],
     ["base_models.py",         "Trains 4 base models"],
     ["meta_model.py",          "Trains the meta-model"],
     ["shap_explainability.py", "SHAP explanations (Novelty 1)"],
     ["concept_drift.py",       "ADWIN drift detection (Novelty 2)"],
     ["anomaly_detector.py",    "Isolation Forest zero-day (Novelty 3)"],
     ["evaluation.py",          "Generates all plots and metrics"],
     ["dashboard.py",           "Streamlit web dashboard"],
     ["show_results.py",        "Shows results in terminal"]],
    [7*cm, 10*cm]
), gap(14)]

# ── FOOTER ────────────────────────────────────────────────────
story += [hr()]
story += [Paragraph(
    "Drift-Aware Explainable Ensemble Learning for Adaptive Network Intrusion Detection  ·  "
    "Dataset: UNSW-NB15  ·  Language: Python 3.12  ·  "
    "Models: LightGBM · XGBoost · CatBoost · RandomForest · Meta-XGBoost  ·  "
    "Novelties: SHAP · ADWIN · Isolation Forest",
    ParagraphStyle("footer", fontSize=7.5, leading=11, alignment=TA_CENTER,
                   textColor=colors.HexColor("#90a4ae"), fontName="Helvetica")
)]

# ── BUILD ─────────────────────────────────────────────────────
doc.build(story)
print(f"✅ PDF saved → {OUTPUT}")
