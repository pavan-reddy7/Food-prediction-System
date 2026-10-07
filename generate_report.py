"""
Indian Food Recognition and Information System Using Machine Learning
7-Chapter Project Report Generator
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image, HRFlowable, KeepTogether
)

# ── Palette ────────────────────────────────────────────────
NAVY      = HexColor('#1a1a2e')
SAFFRON   = HexColor('#FF9F43')
LIGHT_GRAY = HexColor('#f5f5f5')
MID_GRAY  = HexColor('#e0e0e0')
DARK_GRAY = HexColor('#555555')
ALT_ROW   = HexColor('#EFF3FA')

PAGE_W, PAGE_H = A4


# ── Running header / footer ────────────────────────────────
def on_page(canv, doc):
    page = doc.page
    canv.saveState()

    # top accent line
    canv.setStrokeColor(SAFFRON)
    canv.setLineWidth(3)
    canv.line(72, PAGE_H - 36, PAGE_W - 72, PAGE_H - 36)

    if page > 1:
        canv.setFont('Helvetica', 8)
        canv.setFillColor(DARK_GRAY)
        canv.drawRightString(
            PAGE_W - 72, PAGE_H - 52,
            "Indian Food Recognition and Information System Using Machine Learning"
        )

    # footer page number
    canv.setFont('Helvetica', 9)
    canv.setFillColor(DARK_GRAY)
    canv.drawCentredString(PAGE_W / 2, 40, str(page))
    canv.setStrokeColor(MID_GRAY)
    canv.setLineWidth(0.5)
    canv.line(72, 54, PAGE_W - 72, 54)
    canv.restoreState()


# ── Paragraph styles ───────────────────────────────────────
def make_styles():
    def ps(name, **kw):
        return ParagraphStyle(name, **kw)

    return dict(
        cover_title=ps('CoverTitle',
            fontName='Helvetica-Bold', fontSize=22, leading=30,
            alignment=TA_CENTER, textColor=NAVY, spaceAfter=10),
        cover_sub=ps('CoverSub',
            fontName='Helvetica', fontSize=13, leading=18,
            alignment=TA_CENTER, textColor=DARK_GRAY, spaceAfter=8),
        cover_center=ps('CoverCenter',
            fontName='Helvetica', fontSize=11, leading=16,
            alignment=TA_CENTER, textColor=black, spaceAfter=6),
        page_title=ps('PageTitle',
            fontName='Helvetica-Bold', fontSize=18, leading=24,
            alignment=TA_CENTER, textColor=NAVY, spaceBefore=0, spaceAfter=16),
        ch_heading=ps('ChHeading',
            fontName='Helvetica-Bold', fontSize=15, leading=22,
            alignment=TA_LEFT, textColor=NAVY, spaceBefore=6, spaceAfter=8),
        sec_heading=ps('SecHeading',
            fontName='Helvetica-Bold', fontSize=12, leading=17,
            alignment=TA_LEFT, textColor=NAVY, spaceBefore=8, spaceAfter=5),
        body=ps('Body',
            fontName='Helvetica', fontSize=10.5, leading=16,
            alignment=TA_JUSTIFY, spaceAfter=8),
        bullet=ps('Bullet',
            fontName='Helvetica', fontSize=10.5, leading=16,
            alignment=TA_JUSTIFY, spaceAfter=4,
            leftIndent=16, firstLineIndent=-12),
        caption=ps('Caption',
            fontName='Helvetica', fontSize=9, leading=13,
            alignment=TA_CENTER, textColor=DARK_GRAY, spaceAfter=6),
        note=ps('Note',
            fontName='Helvetica', fontSize=10, leading=14,
            alignment=TA_JUSTIFY, textColor=DARK_GRAY,
            spaceAfter=6, leftIndent=12),
        italic_body=ps('ItalicBody',
            fontName='Helvetica-Oblique', fontSize=10.5,
            leading=16, alignment=TA_JUSTIFY,
            leftIndent=20, spaceAfter=8),
    )


# ── Table helpers ──────────────────────────────────────────
def tbl_style():
    return TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0),  NAVY),
        ('TEXTCOLOR',     (0, 0), (-1, 0),  white),
        ('FONTNAME',      (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0),  9.5),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME',      (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',      (0, 1), (-1, -1), 9.5),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [white, ALT_ROW]),
        ('GRID',          (0, 0), (-1, -1), 0.5, MID_GRAY),
        ('LINEBELOW',     (0, 0), (-1, 0),  1.5, SAFFRON),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING',    (0, 0), (-1, -1), 7),
        ('LEFTPADDING',   (0, 0), (-1, -1), 8),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 8),
    ])


def make_table(data, col_widths):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(tbl_style())
    return t


def HR():
    return HRFlowable(width='100%', thickness=0.5,
                      color=MID_GRAY, spaceAfter=6, spaceBefore=2)


def SP(h=0.15):
    return Spacer(1, h * inch)


# ══════════════════════════════════════════════════════════
def build_report():
    pdf_path = r"C:\ML_project\Indian_Food_Recognition_Project_Report.pdf"
    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        rightMargin=72, leftMargin=72,
        topMargin=80, bottomMargin=72,
        title="Indian Food Recognition Report",
        author="Student"
    )

    S = make_styles()
    story = []

    # ══════════════════════════════════════════════════════
    # COVER PAGE
    # ══════════════════════════════════════════════════════
    story += [
        SP(2.0),
        Paragraph("A Project Report on", S['cover_sub']),
        SP(0.1),
        Paragraph(
            "Indian Food Recognition and Information System<br/>Using Machine Learning",
            S['cover_title']),
        SP(0.2),
        HRFlowable(width='60%', thickness=2, color=SAFFRON,
                   hAlign='CENTER', spaceAfter=18),
        Paragraph("Submitted in partial fulfillment of the requirements for the degree of",
                  S['cover_center']),
        Paragraph("<b>Bachelor of Engineering / Technology</b>", S['cover_center']),
        Paragraph("in", S['cover_center']),
        Paragraph("<b>Computer Science and Engineering</b>", S['cover_center']),
        SP(0.3),
        Paragraph("Submitted by", S['cover_sub']),
        Paragraph("<b>[STUDENT NAME]</b>", S['cover_center']),
        Paragraph("Roll No: [ROLL NUMBER]", S['cover_center']),
        SP(0.2),
        Paragraph("Under the guidance of", S['cover_sub']),
        Paragraph("<b>[GUIDE NAME]</b>", S['cover_center']),
        Paragraph("Assistant Professor / Associate Professor", S['cover_center']),
        SP(0.4),
        HRFlowable(width='60%', thickness=1, color=MID_GRAY,
                   hAlign='CENTER', spaceAfter=16),
        Paragraph("<b>Department of Computer Science and Engineering</b>", S['cover_center']),
        Paragraph("<b>[INSTITUTION NAME]</b>", S['cover_center']),
        Paragraph("[CITY, STATE]", S['cover_center']),
        Paragraph("<b>Academic Year: 2025–2026</b>", S['cover_center']),
        PageBreak(),
    ]

    # ══════════════════════════════════════════════════════
    # CERTIFICATE
    # ══════════════════════════════════════════════════════
    story += [
        Paragraph("CERTIFICATE", S['page_title']),
        HR(), SP(0.2),
        Paragraph(
            "This is to certify that the project entitled <b>'Indian Food Recognition and "
            "Information System Using Machine Learning'</b> is a bonafide work carried out by "
            "<b>[STUDENT NAME]</b> (Roll No: [ROLL NUMBER]) in partial fulfillment of the "
            "requirements for the award of the degree of <b>Bachelor of Engineering / Technology "
            "in Computer Science and Engineering</b> from <b>[INSTITUTION NAME]</b> during the "
            "academic year <b>2025–2026</b>.",
            S['body']),
        Paragraph(
            "This project work has been carried out under my guidance and supervision and is "
            "found to be worthy of submission.",
            S['body']),
        SP(1.5),
    ]
    sig_data = [
        ["Guide / Supervisor",  "Head of Department"],
        ["[GUIDE NAME]",        "[HOD NAME]"],
        ["[Designation]",       "[Designation]"],
        ["Department of CSE",   "Department of CSE"],
        ["[Institution Name]",  "[Institution Name]"],
    ]
    sig_t = Table(sig_data, colWidths=[220, 220])
    sig_t.setStyle(TableStyle([
        ('ALIGN',    (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('LINEABOVE',(0, 0), (-1, 0),  1, NAVY),
    ]))
    story += [
        sig_t, SP(1.0),
        Paragraph("Place: ________________________     Date: ________________________",
                  S['cover_center']),
        PageBreak(),
    ]

    # ══════════════════════════════════════════════════════
    # DECLARATION
    # ══════════════════════════════════════════════════════
    story += [
        Paragraph("DECLARATION", S['page_title']),
        HR(), SP(0.2),
        Paragraph(
            "I hereby declare that the project entitled <b>'Indian Food Recognition and "
            "Information System Using Machine Learning'</b>, submitted to the Department of "
            "Computer Science and Engineering, <b>[Institution Name]</b>, in partial fulfillment "
            "of the requirements for the award of the degree of Bachelor of Engineering / "
            "Technology, is my original work and has not previously formed the basis for the "
            "award of any degree, diploma, fellowship, or any other similar title.",
            S['body']),
        Paragraph(
            "The project has been carried out by me under the guidance of <b>[GUIDE NAME]</b>, "
            "[Designation], Department of Computer Science and Engineering. All information "
            "provided in this report is true and correct to the best of my knowledge.",
            S['body']),
        SP(1.5),
        Paragraph("Place: ________________________", S['body']),
        Paragraph("Date:  ________________________", S['body']),
        SP(0.5),
        Paragraph("<b>[STUDENT NAME]</b>", S['body']),
        Paragraph("Roll No: [ROLL NUMBER]", S['body']),
        PageBreak(),
    ]

    # ══════════════════════════════════════════════════════
    # ACKNOWLEDGEMENT
    # ══════════════════════════════════════════════════════
    story += [
        Paragraph("ACKNOWLEDGEMENT", S['page_title']),
        HR(), SP(0.2),
        Paragraph(
            "I would like to express my deepest gratitude to my project guide, "
            "<b>[GUIDE NAME]</b>, [Designation], Department of Computer Science and Engineering, "
            "[Institution Name], for their invaluable guidance, continuous encouragement, and "
            "constructive feedback throughout this project. Their expertise in machine learning "
            "and computer vision greatly shaped the direction of this work.",
            S['body']),
        Paragraph(
            "I extend my sincere thanks to the <b>Head of the Department, [HOD Name]</b>, and "
            "the faculty members of the Department of Computer Science and Engineering for "
            "providing the necessary academic resources and a supportive learning environment.",
            S['body']),
        Paragraph(
            "I am also grateful to the institution for providing access to GPU computing "
            "resources — specifically the <b>NVIDIA RTX 4050</b> — which was instrumental in "
            "training the deep learning models used in this project.",
            S['body']),
        Paragraph(
            "Finally, I thank my family and peers for their constant moral support throughout "
            "this project.",
            S['body']),
        SP(1.0),
        Paragraph("<b>[STUDENT NAME]</b>", S['body']),
        PageBreak(),
    ]

    # ══════════════════════════════════════════════════════
    # ABSTRACT
    # ══════════════════════════════════════════════════════
    story += [
        Paragraph("ABSTRACT", S['page_title']),
        HR(), SP(0.2),
        Paragraph(
            "This project presents an <b>Indian Food Recognition and Information System</b> "
            "designed to classify images of Indian food into 80 categories using deep learning "
            "and transfer learning. The system was built and evaluated through three experimental "
            "phases.",
            S['body']),
        Paragraph(
            "In Phase 1, handcrafted features — RGB color statistics, HSV color statistics, "
            "Gray-Level Co-occurrence Matrix (GLCM) texture descriptors, and Histogram of "
            "Oriented Gradients (HOG) — were extracted and evaluated with classical classifiers "
            "(SVM, Random Forest, KNN), achieving a maximum validation accuracy of ~27%.",
            S['body']),
        Paragraph(
            "In Phase 2, a pretrained ResNet50 model was used as a fixed feature extractor, "
            "producing 2,048-dimensional deep feature vectors classified by SVM and Logistic "
            "Regression, improving accuracy to ~58.5%.",
            S['body']),
        Paragraph(
            "In Phase 3, end-to-end progressive fine-tuning of ResNet50 was performed across "
            "three training phases, incorporating data augmentation, Dropout, label smoothing, "
            "AdamW, cosine annealing, and mixed-precision training on an NVIDIA RTX 4050 GPU. "
            "The final model achieved a validation accuracy of <b>66.75%</b>, Macro Precision "
            "of 0.685, Macro Recall of 0.667, and Macro F1-score of 0.663.",
            S['body']),
        Paragraph(
            "A Streamlit web application was developed for real-time inference, providing "
            "predicted food class, confidence score, Top-5 predictions, and contextual food "
            "information.",
            S['body']),
        Paragraph(
            "<b>Keywords:</b> Indian food recognition, image classification, transfer learning, "
            "ResNet50, fine-tuning, deep learning, PyTorch, Streamlit, computer vision.",
            S['note']),
        PageBreak(),
    ]

    # ══════════════════════════════════════════════════════
    # TABLE OF CONTENTS
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("TABLE OF CONTENTS", S['page_title']))
    story.append(HR())
    story.append(SP(0.1))

    toc_items = [
        ("Certificate",                                      "ii"),
        ("Declaration",                                      "iii"),
        ("Acknowledgement",                                  "iv"),
        ("Abstract",                                         "v"),
        ("List of Figures",                                  "vii"),
        ("List of Tables",                                   "vii"),
        ("Chapter 1: Introduction",                          "1"),
        ("    1.1  Background",                              "1"),
        ("    1.2  Problem Statement",                       "1"),
        ("    1.3  Objectives",                              "2"),
        ("    1.4  Scope",                                   "2"),
        ("Chapter 2: Dataset and Preprocessing",             "3"),
        ("    2.1  Dataset Description",                     "3"),
        ("    2.2  Dataset Statistics",                      "3"),
        ("    2.3  Train / Validation Split",                "3"),
        ("    2.4  Image Preprocessing",                     "4"),
        ("    2.5  Data Augmentation",                       "4"),
        ("Chapter 3: Methodology",                           "5"),
        ("    3.1  System Architecture",                     "5"),
        ("    3.2  Handcrafted Features",                    "6"),
        ("    3.3  Classical ML Algorithms",                 "6"),
        ("    3.4  CNN Feature Extraction",                  "7"),
        ("    3.5  ResNet50",                                "7"),
        ("    3.6  Transfer Learning",                       "8"),
        ("    3.7  Fine-Tuning Strategy",                    "8"),
        ("Chapter 4: Model Training and Experiments",        "9"),
        ("    4.1  Traditional ML Experiments",              "9"),
        ("    4.2  CNN Feature Extraction Experiments",      "10"),
        ("    4.3  ResNet50 Training",                       "10"),
        ("    4.4  Training Configuration",                  "11"),
        ("Chapter 5: Results and Evaluation",                "12"),
        ("    5.1  Model Comparison",                        "12"),
        ("    5.2  Accuracy",                                "13"),
        ("    5.3  Precision, Recall, F1-score",             "13"),
        ("    5.4  Confusion Matrix",                        "13"),
        ("    5.5  Results Discussion",                      "14"),
        ("Chapter 6: Application and Implementation",        "15"),
        ("    6.1  Technology Stack",                        "15"),
        ("    6.2  Streamlit Application",                   "15"),
        ("    6.3  Image Upload",                            "16"),
        ("    6.4  Prediction and Confidence Score",         "16"),
        ("    6.5  Top-5 Predictions",                       "16"),
        ("    6.6  Food Information Module",                 "16"),
        ("Chapter 7: Conclusion and Future Work",            "17"),
        ("    7.1  Conclusion",                              "17"),
        ("    7.2  Limitations",                             "17"),
        ("    7.3  Future Enhancements",                     "18"),
        ("References",                                       "19"),
    ]

    toc_data = []
    chapter_keys = {"Certificate","Declaration","Acknowledgement","Abstract",
                    "List of Figures","List of Tables","References"}
    for entry, pg in toc_items:
        is_ch = entry.strip().startswith("Chapter") or entry.strip() in chapter_keys
        row = [
            Paragraph(
                f"<font name='Helvetica-Bold'>{entry}</font>" if is_ch else entry,
                ParagraphStyle('te', fontName='Helvetica', fontSize=10, leading=14)),
            Paragraph(
                f"<font name='Helvetica-Bold'>{pg}</font>" if is_ch else pg,
                ParagraphStyle('tp', fontName='Helvetica', fontSize=10, leading=14,
                               alignment=TA_RIGHT)),
        ]
        toc_data.append(row)

    toc_t = Table(toc_data, colWidths=[380, 60])
    toc_t.setStyle(TableStyle([
        ('ALIGN',         (1, 0), (1, -1), 'RIGHT'),
        ('LINEBELOW',     (0, 0), (-1, -1), 0.3, MID_GRAY),
        ('TOPPADDING',    (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(toc_t)
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # LIST OF FIGURES & TABLES  (single page)
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("LIST OF FIGURES", S['page_title']))
    story.append(HR())
    lof_data = [
        ["Figure No.", "Title",                                              "Page"],
        ["Figure 1",  "System Architecture — Training Pipeline",            "5"],
        ["Figure 2",  "System Architecture — Inference Pipeline",           "5"],
        ["Figure 3",  "ResNet50 Three-Phase Fine-Tuning Workflow",          "8"],
        ["Figure 4",  "Confusion Matrix — Validation Set (80 Classes)",     "13"],
        ["Figure 5",  "Streamlit Application — Prediction Results Screen",  "15"],
    ]
    story.append(make_table(lof_data, [70, 320, 50]))
    story.append(SP(0.3))

    story.append(Paragraph("LIST OF TABLES", S['page_title']))
    story.append(HR())
    lot_data = [
        ["Table No.", "Title",                                    "Page"],
        ["Table 1",  "Dataset Statistics",                       "3"],
        ["Table 2",  "Train / Validation Split",                 "3"],
        ["Table 3",  "Handcrafted Feature Summary",              "6"],
        ["Table 4",  "System Architecture — Training Pipeline",  "5"],
        ["Table 5",  "System Architecture — Inference Pipeline", "5"],
        ["Table 6",  "Traditional ML Experiment Results",        "9"],
        ["Table 7",  "CNN Feature Extraction Results",           "10"],
        ["Table 8",  "Three-Phase Training Configuration",       "11"],
        ["Table 9",  "Data Augmentation Techniques",             "11"],
        ["Table 10", "Final Model Evaluation Metrics",           "13"],
        ["Table 11", "Overall Model Comparison",                 "12"],
        ["Table 12", "Technology Stack",                         "15"],
    ]
    story.append(make_table(lot_data, [70, 320, 50]))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # CHAPTER 1 — INTRODUCTION
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("Chapter 1: Introduction", S['ch_heading']))
    story.append(HR())

    story.append(Paragraph("1.1  Background", S['sec_heading']))
    story += [
        Paragraph(
            "Food recognition from images is an active research area in computer vision with "
            "applications spanning dietary monitoring, restaurant automation, nutritional "
            "analysis, and cultural heritage documentation. The widespread availability of "
            "smartphone cameras and open-source deep learning frameworks has accelerated "
            "progress in automated food classification systems.",
            S['body']),
        Paragraph(
            "Indian cuisine is one of the most diverse culinary traditions in the world, "
            "comprising hundreds of distinct dishes that vary significantly across regions, "
            "communities, and preparation styles. Dishes often share similar visual appearances "
            "— for example, different lentil-based gravies or flatbreads — making automated "
            "fine-grained classification particularly challenging.",
            S['body']),
        Paragraph(
            "Traditional handcrafted image features (color histograms, texture descriptors) "
            "have proven insufficient for such tasks. The development of deep convolutional "
            "neural networks (CNNs), and specifically ResNet50 (He et al., 2016), enables "
            "state-of-the-art performance by learning hierarchical representations from "
            "raw pixel data.",
            S['body']),
    ]

    story.append(Paragraph("1.2  Problem Statement", S['sec_heading']))
    story += [
        Paragraph(
            "Given an RGB image of an Indian food item, automatically predict the correct "
            "food category from a set of 80 possible classes with high accuracy. The specific "
            "challenges include:",
            S['body']),
        Paragraph("• <b>Fine-grained similarity:</b> Many Indian dishes share similar color, texture, and shape (e.g., various dals, chutneys, rice dishes).", S['bullet']),
        Paragraph("• <b>Limited dataset size:</b> Only approximately 50 images per class — small for deep learning.", S['bullet']),
        Paragraph("• <b>High intra-class variation:</b> The same dish looks different across regions, lighting, and plating.", S['bullet']),
        Paragraph("• <b>Generalization:</b> Classifying real-world images that may differ from the training distribution.", S['bullet']),
    ]

    story.append(Paragraph("1.3  Objectives", S['sec_heading']))
    story += [
        Paragraph("1. Preprocess an Indian food image dataset of 80 classes (~4,000 images).", S['bullet']),
        Paragraph("2. Evaluate handcrafted features (RGB, HSV, GLCM, HOG) with classical ML classifiers as a baseline.", S['bullet']),
        Paragraph("3. Evaluate pretrained ResNet50 as a fixed CNN feature extractor with downstream classifiers.", S['bullet']),
        Paragraph("4. Implement three-phase progressive fine-tuning of ResNet50 with advanced regularization.", S['bullet']),
        Paragraph("5. Quantitatively compare all approaches using accuracy, macro precision, recall, and F1-score.", S['bullet']),
        Paragraph("6. Deploy a functional Streamlit web application for real-time food image classification.", S['bullet']),
        Paragraph("7. Provide contextual food information (category, ingredients, description) for each predicted class.", S['bullet']),
    ]

    story.append(Paragraph("1.4  Scope", S['sec_heading']))
    story += [
        Paragraph("• The system classifies images into one of 80 predefined Indian food categories. Object detection and ingredient recognition from pixels are outside scope.", S['bullet']),
        Paragraph("• Food information is retrieved from a static class-to-information dictionary, not inferred from the image.", S['bullet']),
        Paragraph("• The model is trained and evaluated on a fixed dataset; online learning or continuous updates are not included.", S['bullet']),
        Paragraph("• Deployment is a local Streamlit server. Cloud deployment is a future enhancement.", S['bullet']),
        Paragraph("• The project is implemented in Python 3, PyTorch, on a Windows machine with NVIDIA RTX 4050 GPU.", S['bullet']),
        Paragraph("• The application accepts JPG, JPEG, PNG, and WEBP formats.", S['bullet']),
        PageBreak(),
    ]

    # ══════════════════════════════════════════════════════
    # CHAPTER 2 — DATASET AND PREPROCESSING
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("Chapter 2: Dataset and Preprocessing", S['ch_heading']))
    story.append(HR())

    story.append(Paragraph("2.1  Dataset Description", S['sec_heading']))
    story += [
        Paragraph(
            "The dataset used is the <b>Indian Food Images Dataset</b>, a curated collection "
            "of food images representing 80 distinct Indian food categories. The dataset is "
            "organized in a folder-per-class structure, with each folder containing images "
            "of a single food class.",
            S['body']),
        Paragraph(
            "The 80 food classes span Indian regional cuisines, covering main courses, breads, "
            "sweets, snacks, and beverages. Representative classes include: Biryani, Butter "
            "Chicken, Dal Tadka, Palak Paneer, Chapati, Naan, Jalebi, Gulab Jamun, Rasgulla, "
            "Modak, Poha, Lassi, Mysore Pak, Adhirasam, Pootharekulu, and 65 additional classes.",
            S['body']),
    ]

    story.append(Paragraph("2.2  Dataset Statistics", S['sec_heading']))
    ds_data = [
        ["Parameter",              "Value"],
        ["Dataset Name",           "Indian Food Images Dataset"],
        ["Total Images",           "~4,000"],
        ["Number of Classes",      "80"],
        ["Images per Class (avg)", "~50"],
        ["Image Format",           "JPG / PNG"],
        ["Organization",           "Folder-per-class (ImageFolder structure)"],
    ]
    story.append(make_table(ds_data, [220, 220]))
    story += [SP(0.1), Paragraph("Table 1: Dataset Statistics", S['caption'])]

    story.append(Paragraph("2.3  Train / Validation Split", S['sec_heading']))
    split_data = [
        ["Subset",          "Approximate Images", "Percentage"],
        ["Training Set",    "~3,200 images",       "80%"],
        ["Validation Set",  "~800 images",          "20%"],
        ["Total",           "~4,000 images",        "100%"],
    ]
    story.append(make_table(split_data, [180, 180, 100]))
    story += [
        SP(0.1),
        Paragraph("Table 2: Train / Validation Split", S['caption']),
        Paragraph(
            "A stratified 80/20 split with a fixed random seed (42) was used across all "
            "training phases, ensuring that the validation set is never seen during training "
            "and that results are reproducible.",
            S['note']),
    ]

    story.append(Paragraph("2.4  Image Preprocessing", S['sec_heading']))
    story += [
        Paragraph(
            "Two separate preprocessing pipelines were used — one deterministic pipeline "
            "for the validation set, and one stochastic pipeline (with augmentation) for "
            "the training set.",
            S['body']),
        Paragraph("<b>Validation preprocessing (deterministic):</b>", S['body']),
        Paragraph("1. <b>Resize(256):</b> Resize the shorter side to 256 pixels, maintaining aspect ratio.", S['bullet']),
        Paragraph("2. <b>CenterCrop(224):</b> Crop the central 224×224 pixel region.", S['bullet']),
        Paragraph("3. <b>ToTensor:</b> Convert PIL Image to PyTorch tensor, values in [0, 1].", S['bullet']),
        Paragraph("4. <b>Normalize:</b> Subtract ImageNet channel means [0.485, 0.456, 0.406] and divide by std [0.229, 0.224, 0.225].", S['bullet']),
    ]

    story.append(Paragraph("2.5  Data Augmentation", S['sec_heading']))
    story += [
        Paragraph(
            "Training images underwent the following stochastic augmentation pipeline to "
            "improve generalization by exposing the model to diverse visual variations:",
            S['body']),
    ]
    aug_data = [
        ["Augmentation Technique",  "Parameters",                                         "Purpose"],
        ["RandomResizedCrop",       "size=224, scale=(0.8, 1.0)",                         "Scale/Crop invariance"],
        ["RandomHorizontalFlip",    "p=0.5",                                              "Orientation invariance"],
        ["RandomRotation",          "degrees=15",                                         "Rotation robustness"],
        ["ColorJitter",             "brightness=0.3, contrast=0.3, sat=0.3, hue=0.05",   "Lighting invariance"],
        ["TrivialAugmentWide",      "Auto-selected policy",                               "Diverse augmentation pool"],
        ["RandomErasing",           "p=0.25, scale=(0.02, 0.20)",                         "Occlusion robustness"],
    ]
    story.append(make_table(aug_data, [145, 185, 110]))
    story += [
        SP(0.1),
        Paragraph("Table 9: Data Augmentation Techniques", S['caption']),
        Paragraph(
            "Note: All augmentations are applied stochastically per batch during training only. "
            "The validation set is never augmented.",
            S['note']),
        PageBreak(),
    ]

    # ══════════════════════════════════════════════════════
    # CHAPTER 3 — METHODOLOGY
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("Chapter 3: Methodology", S['ch_heading']))
    story.append(HR())

    story.append(Paragraph("3.1  System Architecture", S['sec_heading']))
    story += [
        Paragraph(
            "The system has two main pipelines: the <b>Training Pipeline</b> and the "
            "<b>Inference / Application Pipeline</b>.",
            S['body']),
        Paragraph("<b>Training Pipeline:</b>", S['body']),
    ]
    arch_data = [
        ["Step", "Component",           "Details"],
        ["1",    "Raw Images",          "~4,000 images, 80 classes, folder-per-class"],
        ["2",    "Preprocessing",       "Resize → CenterCrop(224) → Normalize (ImageNet stats)"],
        ["3",    "Augmentation",        "RandomCrop, Flip, Rotation, ColorJitter, Erasing (train only)"],
        ["4",    "ResNet50 Backbone",   "Pretrained on ImageNet — 48 conv layers + BN + ReLU"],
        ["5",    "Custom Head",         "GlobalAvgPool(2048) → Dropout(0.5) → Linear(80)"],
        ["6",    "Loss / Optimizer",    "CrossEntropyLoss (LS=0.1) + AdamW + CosineAnnealingLR"],
        ["7",    "Saved Checkpoint",    ".pth file with model_state_dict + class list"],
    ]
    story.append(make_table(arch_data, [35, 130, 275]))
    story += [SP(0.1), Paragraph("Table 4: System Architecture — Training Pipeline", S['caption']), SP(0.1)]

    story += [Paragraph("<b>Inference / Application Pipeline:</b>", S['body'])]
    infer_data = [
        ["Step", "Component",                "Output"],
        ["1",    "User Upload (Streamlit)",  "Raw image file (JPG/PNG/WEBP)"],
        ["2",    "PIL Image Loading",        "RGB Image object"],
        ["3",    "Preprocessing Transform",  "Tensor (1 × 3 × 224 × 224)"],
        ["4",    "ResNet50 Forward Pass",    "Logits (1 × 80)"],
        ["5",    "Softmax + Top-K",          "Top-5 classes with probabilities"],
        ["6",    "Food Info Lookup",         "Category, Ingredients, Description (class-mapped)"],
        ["7",    "Display (Streamlit)",      "Prediction card + food info panel"],
    ]
    story.append(make_table(infer_data, [35, 160, 245]))
    story += [SP(0.1), Paragraph("Table 5: System Architecture — Inference Pipeline", S['caption'])]

    story.append(Paragraph("3.2  Handcrafted Features", S['sec_heading']))
    story += [
        Paragraph(
            "In Phase 1, the following handcrafted features were extracted from each image:",
            S['body']),
    ]
    feat_data = [
        ["Feature Type",   "Description",                                                                  "Dim"],
        ["RGB Color Stats","Mean and standard deviation of each of the 3 RGB channels",                    "6"],
        ["HSV Color Stats","Mean and standard deviation of H, S, V channels",                              "6"],
        ["GLCM Texture",   "Contrast, Correlation, Energy, Homogeneity computed on grayscale (16 levels)", "4"],
        ["HOG Features",   "Histogram of Oriented Gradients: 9 orientations, 16×16 px/cell, 2×2 cells/block","Variable"],
    ]
    story.append(make_table(feat_data, [100, 270, 70]))
    story += [SP(0.1), Paragraph("Table 3: Handcrafted Feature Summary", S['caption'])]

    story.append(Paragraph("3.3  Classical ML Algorithms", S['sec_heading']))
    story += [
        Paragraph(
            "<b>Support Vector Machine (SVM):</b> Finds the optimal hyperplane maximizing the "
            "margin between classes. Two kernel variants were evaluated: Linear SVM "
            "(computationally efficient on high-dimensional data) and RBF SVM (non-linear "
            "decision boundaries via a Gaussian kernel).",
            S['body']),
        Paragraph(
            "<b>Random Forest:</b> Ensemble of decision trees trained on random subsets of "
            "data and features, aggregating predictions by majority vote. Robust to overfitting.",
            S['body']),
        Paragraph(
            "<b>K-Nearest Neighbors (KNN):</b> Classifies a new point by majority class among "
            "its K nearest neighbors. A non-parametric baseline method.",
            S['body']),
        Paragraph(
            "<b>Logistic Regression:</b> Models class probabilities using a softmax over a "
            "linear combination of input features. Simple, interpretable multi-class baseline.",
            S['body']),
    ]

    story.append(Paragraph("3.4  CNN Feature Extraction", S['sec_heading']))
    story += [
        Paragraph(
            "In Phase 2, a pretrained ResNet50 model was used as a fixed feature extractor. "
            "The final classification layer was removed, and the output of the global average "
            "pooling layer (a 2,048-dimensional vector) was used as the image representation. "
            "These vectors were extracted for all images and saved as a CSV feature matrix. "
            "Downstream classical classifiers (Linear SVM, RBF SVM, Logistic Regression) were "
            "trained on these features without any fine-tuning of the backbone.",
            S['body']),
    ]

    story.append(Paragraph("3.5  ResNet50", S['sec_heading']))
    story += [
        Paragraph(
            "ResNet50 is a 50-layer deep residual convolutional neural network. It consists of "
            "an initial 7×7 convolutional layer, max pooling, then four residual groups "
            "(layer1–layer4) each containing bottleneck blocks with skip connections "
            "(1×1 → 3×3 → 1×1 convolutions + Batch Normalization + ReLU), and finally global "
            "average pooling.",
            S['body']),
        Paragraph(
            "In this project, the original 1,000-class ImageNet head was replaced with: "
            "<b>Dropout(0.5) → Linear(2048, 80)</b>.",
            S['body']),
    ]

    story.append(Paragraph("3.6  Transfer Learning", S['sec_heading']))
    story += [
        Paragraph(
            "Transfer learning initializes the model with weights pretrained on ImageNet "
            "(~1.2M images, 1,000 classes) and then fine-tunes on the target dataset "
            "(Indian Food, ~4,000 images, 80 classes). Early convolutional layers learn "
            "generic features (edges, textures); deeper layers learn task-specific semantics. "
            "By reusing generic lower-layer features, we significantly reduce the data "
            "requirement and training time.",
            S['body']),
    ]

    story.append(Paragraph("3.7  Fine-Tuning Strategy", S['sec_heading']))
    story += [
        Paragraph(
            "A three-phase progressive fine-tuning strategy was adopted to balance learning "
            "speed, stability, and final accuracy:",
            S['body']),
    ]
    phase_data = [
        ["Phase",    "Unfrozen Layers",                    "Epochs", "Learning Rate", "Scheduler"],
        ["Phase 1",  "Classifier head (fc) only",          "5",      "1e-3",          "None"],
        ["Phase 2",  "layer4 + classifier head",           "≤20",    "5e-5",          "CosineAnnealingLR"],
        ["Phase 3",  "layer3 + layer4 + classifier head",  "≤20",    "1e-5",          "CosineAnnealingLR"],
    ]
    story.append(make_table(phase_data, [50, 165, 55, 100, 110]))
    story += [
        SP(0.1),
        Paragraph("Table 8: Three-Phase Training Configuration", S['caption']),
        Paragraph(
            "<b>Phase 1</b> trains only the new head while the backbone is frozen, quickly "
            "adapting the output layer to predict Indian food classes. "
            "<b>Phase 2</b> unfreezes layer4, allowing the highest-level semantic features "
            "to adapt, at a low learning rate (5e-5) to avoid destroying pretrained "
            "representations. "
            "<b>Phase 3</b> extends fine-tuning to layer3 as well, with an even lower "
            "learning rate (1e-5). Early stopping (patience=8) is applied in Phases 2 and 3.",
            S['body']),
        PageBreak(),
    ]

    # ══════════════════════════════════════════════════════
    # CHAPTER 4 — MODEL TRAINING AND EXPERIMENTS
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("Chapter 4: Model Training and Experiments", S['ch_heading']))
    story.append(HR())

    story.append(Paragraph("4.1  Traditional ML Experiments", S['sec_heading']))
    story += [
        Paragraph(
            "Phase 1 evaluated handcrafted image features with classical classifiers. "
            "Images were preprocessed to a uniform size; RGB, HSV, GLCM, and HOG features "
            "were extracted and combined into a single feature vector per image. "
            "The same 80/20 stratified split was used throughout.",
            S['body']),
    ]
    trad_data = [
        ["Feature Set",                  "Classifier",     "Validation Accuracy"],
        ["RGB + HSV + GLCM + HOG",       "KNN",            "~22%"],
        ["RGB + HSV + GLCM + HOG",       "Linear SVM",     "~24%"],
        ["RGB + HSV + GLCM + HOG",       "Random Forest",  "~25%"],
        ["RGB + HSV + GLCM + HOG",       "RBF SVM",        "~27%"],
    ]
    story.append(make_table(trad_data, [200, 140, 100]))
    story += [
        SP(0.1),
        Paragraph("Table 6: Traditional ML Experiment Results", S['caption']),
        Paragraph(
            "The best result — ~27% with RBF SVM — is far below acceptable performance. "
            "Handcrafted features capture only low-level image statistics and fail to model "
            "the complex, hierarchical visual patterns that distinguish fine-grained food "
            "categories. This confirms that deep learning representations are necessary.",
            S['body']),
    ]

    story.append(Paragraph("4.2  CNN Feature Extraction Experiments", S['sec_heading']))
    story += [
        Paragraph(
            "Phase 2 replaced handcrafted features with 2,048-dimensional deep features "
            "from a frozen ResNet50 backbone pretrained on ImageNet. Classical classifiers "
            "were trained on these features.",
            S['body']),
    ]
    cnn_data = [
        ["Feature Extractor",    "Classifier",           "Validation Accuracy"],
        ["ResNet50 (frozen)",    "Logistic Regression",  "~52%"],
        ["ResNet50 (frozen)",    "Linear SVM",           "~54%"],
        ["ResNet50 (frozen)",    "RBF SVM",              "~58.5%"],
    ]
    story.append(make_table(cnn_data, [200, 140, 100]))
    story += [
        SP(0.1),
        Paragraph("Table 7: CNN Feature Extraction Experiment Results", S['caption']),
        Paragraph(
            "Switching from handcrafted to CNN features yielded a dramatic improvement of "
            "~31 percentage points (from 27% to 58.5%). This confirms that ImageNet "
            "pretrained representations are highly transferable to the Indian food domain, "
            "even without task-specific adaptation. The RBF SVM achieved the best result, "
            "suggesting non-linear class boundaries in the CNN feature space.",
            S['body']),
    ]

    story.append(Paragraph("4.3  ResNet50 Training", S['sec_heading']))
    story += [
        Paragraph(
            "Phase 3 involved end-to-end progressive fine-tuning of ResNet50. Training "
            "proceeded in three phases as described in Section 3.7. The model with the "
            "highest validation accuracy at each phase was saved as a checkpoint (.pth). "
            "Mixed-precision training (torch.amp, float16) was used throughout to reduce "
            "GPU memory and accelerate training on the NVIDIA RTX 4050.",
            S['body']),
    ]

    story.append(Paragraph("4.4  Training Configuration", S['sec_heading']))
    config_data = [
        ["Hyperparameter / Setting",   "Value"],
        ["Framework",                  "PyTorch"],
        ["GPU",                        "NVIDIA RTX 4050"],
        ["Batch Size",                 "32"],
        ["Input Image Size",           "224 × 224 pixels"],
        ["Optimizer",                  "AdamW"],
        ["Weight Decay (L2 Reg.)",     "0.01"],
        ["Loss Function",              "CrossEntropyLoss"],
        ["Label Smoothing",            "0.1"],
        ["Phase 1 LR / Epochs",        "1e-3 / 5"],
        ["Phase 2 LR / Max Epochs",    "5e-5 / 20"],
        ["Phase 3 LR / Max Epochs",    "1e-5 / 20"],
        ["LR Scheduler (Phase 2 & 3)", "CosineAnnealingLR (eta_min: 1e-6 / 1e-7)"],
        ["Early Stopping Patience",    "8 epochs"],
        ["Mixed Precision",            "Enabled (torch.amp, float16)"],
        ["Random Seed",                "42"],
    ]
    story.append(make_table(config_data, [260, 180]))
    story += [
        SP(0.1),
        Paragraph("Table 8 (detail): Training Configuration", S['caption']),
        Paragraph(
            "<b>Label Smoothing (ε=0.1):</b> Replaces hard target probability 1.0 with 0.9, "
            "distributing the remaining 0.1 across other classes, preventing overconfidence. "
            "<b>AdamW</b> decouples weight decay from the gradient update step, providing "
            "more consistent regularization. <b>CosineAnnealingLR</b> smoothly reduces the "
            "learning rate to a minimum, helping the optimizer settle into a sharper minimum.",
            S['body']),
        PageBreak(),
    ]

    # ══════════════════════════════════════════════════════
    # CHAPTER 5 — RESULTS AND EVALUATION
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("Chapter 5: Results and Evaluation", S['ch_heading']))
    story.append(HR())

    story.append(Paragraph("5.1  Model Comparison", S['sec_heading']))
    comp_data = [
        ["Approach",                          "Classifier / Method",  "Accuracy"],
        ["Handcrafted Features",              "KNN",                  "~22%"],
        ["Handcrafted Features",              "Linear SVM",           "~24%"],
        ["Handcrafted Features",              "Random Forest",        "~25%"],
        ["Handcrafted Features",              "RBF SVM",              "~27%"],
        ["CNN Features (ResNet50, frozen)",   "Logistic Regression",  "~52%"],
        ["CNN Features (ResNet50, frozen)",   "Linear SVM",           "~54%"],
        ["CNN Features (ResNet50, frozen)",   "RBF SVM",              "~58.5%"],
        ["Fine-tuned ResNet50 (End-to-End)", "3-Phase Fine-Tuning",   "66.75%"],
    ]
    story.append(make_table(comp_data, [215, 160, 65]))
    story += [SP(0.1), Paragraph("Table 11: Overall Model Comparison (All Experiments)", S['caption'])]

    story.append(Paragraph("5.2  Accuracy", S['sec_heading']))
    story += [
        Paragraph(
            "The final fine-tuned ResNet50 model achieved a <b>validation accuracy of 66.75%</b> "
            "on the held-out 20% validation set. For context, random chance yields 1/80 = 1.25%. "
            "The progression from 27% → 58.5% → 66.75% demonstrates a clear and consistent "
            "improvement across the three experimental phases.",
            S['body']),
    ]

    story.append(Paragraph("5.3  Precision, Recall, F1-score", S['sec_heading']))
    metrics_data = [
        ["Metric",              "Value"],
        ["Validation Accuracy", "66.75%"],
        ["Macro Precision",     "0.685"],
        ["Macro Recall",        "0.667"],
        ["Macro F1-score",      "0.663"],
    ]
    story.append(make_table(metrics_data, [220, 220]))
    story += [
        SP(0.1),
        Paragraph("Table 10: Final Model Evaluation Metrics", S['caption']),
        Paragraph(
            "Macro metrics give equal weight to each class regardless of class size. "
            "A Macro Precision of 0.685 slightly higher than Macro Recall (0.667) indicates "
            "the model is moderately conservative in its predictions. The Macro F1-score of "
            "0.663 reflects balanced performance across all 80 classes.",
            S['body']),
    ]

    story.append(Paragraph("5.4  Confusion Matrix", S['sec_heading']))
    img_path = r"C:\ML_project\confusion_matrix_v2.png"
    if os.path.exists(img_path):
        story += [SP(0.1), Image(img_path, width=400, height=400), SP(0.1)]
        story.append(Paragraph("Figure 4: Confusion Matrix — Validation Set (80 Classes)", S['caption']))
    else:
        story.append(Paragraph(
            "[NOTE: Confusion matrix image not found. Run evaluate_resnet_v2.py to generate "
            "confusion_matrix_v2.png and re-run this script.]",
            S['note']))
    story += [
        Paragraph(
            "The confusion matrix is an 80×80 grid where diagonal elements represent correct "
            "classifications and off-diagonal elements represent misclassifications. "
            "A strong diagonal with minimal off-diagonal activity indicates good performance.",
            S['body']),
        Paragraph("• Classes with high visual similarity (e.g., different lentil curries) are more often confused with each other.", S['bullet']),
        Paragraph("• Classes with distinctive visual features (e.g., Jalebi's spiral shape, Naan's irregular surface) achieve higher per-class accuracy.", S['bullet']),
        Paragraph("• Classes with fewer effective training samples show higher miss rates.", S['bullet']),
    ]

    story.append(Paragraph("5.5  Results Discussion", S['sec_heading']))
    story += [
        Paragraph(
            "<b>Phase 1 → Phase 2:</b> Replacing handcrafted features with frozen CNN features "
            "yielded a ~31 percentage point improvement (27% → 58.5%). Deep networks learn "
            "qualitatively superior representations for complex fine-grained visual tasks.",
            S['body']),
        Paragraph(
            "<b>Phase 2 → Phase 3:</b> End-to-end fine-tuning added ~8 percentage points "
            "(58.5% → 66.75%), attributable to backbone adaptation to discriminate specifically "
            "between Indian food classes.",
            S['body']),
        Paragraph(
            "<b>Regularization impact:</b> Label smoothing, Dropout, weight decay, and "
            "extensive augmentation were critical in controlling overfitting on the small "
            "dataset (~50 images per class), collectively improving generalization to the "
            "validation set.",
            S['body']),
        Paragraph(
            "A 66.75% accuracy on an 80-class fine-grained task with ~50 images per class "
            "is a meaningful result, substantially above the frozen-backbone ceiling and far "
            "above any handcrafted baseline.",
            S['body']),
        PageBreak(),
    ]

    # ══════════════════════════════════════════════════════
    # CHAPTER 6 — APPLICATION AND IMPLEMENTATION
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("Chapter 6: Application and Implementation", S['ch_heading']))
    story.append(HR())

    story.append(Paragraph("6.1  Technology Stack", S['sec_heading']))
    tech_data = [
        ["Tool / Library",   "Version / Detail",       "Purpose"],
        ["Python",           "3.x",                    "Primary programming language"],
        ["PyTorch",          "Latest stable",           "Deep learning framework"],
        ["torchvision",      "Companion to PyTorch",   "Pretrained models, transforms"],
        ["Streamlit",        "Latest stable",           "Web application framework"],
        ["scikit-learn",     "Latest stable",           "Classical ML classifiers, metrics"],
        ["scikit-image",     "Latest stable",           "GLCM, HOG feature extraction"],
        ["OpenCV (cv2)",     "Latest stable",           "Image loading, color conversion"],
        ["NumPy",            "Latest stable",           "Numerical array operations"],
        ["Pandas",           "Latest stable",           "Feature DataFrame management"],
        ["Matplotlib",       "Latest stable",           "Confusion matrix visualization"],
        ["PIL / Pillow",     "Latest stable",           "Image loading in Streamlit app"],
    ]
    story.append(make_table(tech_data, [115, 135, 190]))
    story += [SP(0.1), Paragraph("Table 12: Technology Stack", S['caption'])]

    story.append(Paragraph("6.2  Streamlit Application", S['sec_heading']))
    story += [
        Paragraph(
            "A complete web application was developed using Streamlit, providing a clean, "
            "dark-themed professional interface for real-time Indian food recognition. "
            "The application is launched via <b>streamlit run app.py</b> and runs locally.",
            S['body']),
        Paragraph(
            "The model is loaded once and cached with <b>@st.cache_resource</b>, preventing "
            "repeated disk reads on each user interaction. The checkpoint file contains "
            "both the model state dictionary and the list of class names.",
            S['body']),
        Paragraph(
            "<b>Key application features:</b>",
            S['body']),
        Paragraph("• <b>Hero Header:</b> Project title, subtitle, and model badges (ResNet-50, 80 Food Classes, Transfer Learning).", S['bullet']),
        Paragraph("• <b>About the AI Model (collapsible):</b> Model metadata and a visual pipeline diagram.", S['bullet']),
        Paragraph("• <b>Analyze Another Image button:</b> Resets the uploader using Streamlit's session state key mechanism.", S['bullet']),
    ]

    story.append(Paragraph("6.3  Image Upload", S['sec_heading']))
    story += [
        Paragraph(
            "The application accepts images in JPG, JPEG, PNG, and WEBP formats via a "
            "drag-and-drop or file browser interface (st.file_uploader). The uploaded image "
            "is displayed in the results panel alongside the prediction.",
            S['body']),
    ]
    upload_example = r"C:\Users\pavan\OneDrive\Pictures\Screenshots\Screenshot 2026-09-15 092724.png"
    if os.path.exists(upload_example):
        story += [SP(0.08), Image(upload_example, width=400, height=188), SP(0.06)]
        story.append(Paragraph("Figure 5: Streamlit upload and high-confidence Naan prediction.", S['caption']))

    story.append(Paragraph("6.4  Prediction and Confidence Score", S['sec_heading']))
    story += [
        Paragraph(
            "After upload, the image is preprocessed (Resize → CenterCrop → ToTensor → Normalize), "
            "passed through the fine-tuned ResNet50, and softmax probabilities are computed "
            "over 80 classes. The top predicted class and its confidence score are displayed "
            "in a styled prediction card.",
            S['body']),
        Paragraph(
            "• If confidence ≥ 50%: A green 'Analysis Complete' result card is shown.",
            S['bullet']),
        Paragraph(
            "• If confidence < 50%: An amber warning advises the user to try a clearer image.",
            S['bullet']),
    ]

    story.append(Paragraph("6.5  Top-5 Predictions", S['sec_heading']))
    story += [
        Paragraph(
            "A collapsible 'Show detailed predictions' section displays the Top-5 ranked "
            "predicted classes with their percentage confidence scores and progress bars. "
            "This allows users to inspect model uncertainty and alternative candidates.",
            S['body']),
    ]
    top5_example = r"C:\Users\pavan\OneDrive\Pictures\Screenshots\Screenshot 2026-09-15 092841.png"
    if os.path.exists(top5_example):
        story += [SP(0.08), Image(top5_example, width=400, height=188), SP(0.06)]
        story.append(Paragraph("Figure 6: Expanded Top-5 predictions for a Biryani image.", S['caption']))

    story.append(Paragraph("6.6  Food Information Module", S['sec_heading']))
    story += [
        Paragraph(
            "An important distinction: the food information displayed — category, common "
            "ingredients, and description — is <b>NOT</b> inferred from the image. "
            "It is retrieved from a static Python dictionary that maps each of the 80 class "
            "names to pre-written structured information. The model classifies the image into "
            "a class name, and that name is used as a lookup key.",
            S['note']),
        Paragraph(
            "Information provided for each class:",
            S['body']),
        Paragraph("• <b>Category:</b> Main Course, Sweet, Breakfast, Beverage, Indian Bread, etc.", S['bullet']),
        Paragraph("• <b>Common Ingredients:</b> Curated list of typical ingredients for the dish.", S['bullet']),
        Paragraph("• <b>About:</b> Concise description of the food item, regional origin, and preparation method.", S['bullet']),
        Paragraph(
            "The dictionary currently covers 15 food classes with detailed information. "
            "For classes not yet in the dictionary, a generic fallback message is shown.",
            S['body']),
        Paragraph(
            "<b>Project file structure:</b> The project is organized into distinct scripts: "
            "preprocess.py (image preprocessing), extract_features.py (handcrafted features), "
            "cnn_features.py (CNN feature extraction), train_svm.py / cnn_svm.py / cnn_logistic.py "
            "(classical classifiers), train_resnet_v2.py (ResNet50 fine-tuning), "
            "evaluate_resnet_v2.py (evaluation + confusion matrix), and app.py (Streamlit app). "
            "The final model checkpoint is best_resnet50_food_gpu_v2_resumed.pth.",
            S['body']),
    ]
    low_confidence_example = r"C:\Users\pavan\OneDrive\Pictures\Screenshots\Screenshot 2026-09-15 093124.png"
    if os.path.exists(low_confidence_example):
        story += [SP(0.08), Image(low_confidence_example, width=400, height=188), SP(0.06)]
        story.append(Paragraph("Figure 7: Low-confidence Butter Chicken prediction warning.", S['caption']))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # CHAPTER 7 — CONCLUSION AND FUTURE WORK
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("Chapter 7: Conclusion and Future Work", S['ch_heading']))
    story.append(HR())

    story.append(Paragraph("7.1  Conclusion", S['sec_heading']))
    story += [
        Paragraph(
            "This project successfully developed an <b>Indian Food Recognition and Information "
            "System</b> capable of classifying images into 80 distinct Indian food categories "
            "using deep learning and transfer learning techniques.",
            S['body']),
        Paragraph(
            "The project followed a systematic experimental progression. Handcrafted features "
            "(RGB, HSV, GLCM, HOG) with classical classifiers yielded a peak accuracy of only "
            "~27%, confirming their inadequacy for fine-grained food recognition. CNN-based "
            "feature extraction using a frozen ResNet50 backbone improved accuracy to ~58.5%, "
            "demonstrating the power of deep learned representations. Finally, end-to-end "
            "progressive fine-tuning of ResNet50 — with data augmentation, label smoothing, "
            "AdamW, cosine annealing, and GPU-accelerated mixed-precision training — achieved "
            "a validation accuracy of <b>66.75%</b> with a Macro F1-score of 0.663.",
            S['body']),
        Paragraph(
            "A fully functional Streamlit web application was developed with real-time "
            "inference, confidence scores, Top-5 predictions, and contextual food information. "
            "The system demonstrates a practical, deployable machine learning solution for "
            "a culturally significant real-world classification problem.",
            S['body']),
    ]

    story.append(Paragraph("7.2  Limitations", S['sec_heading']))
    story += [
        Paragraph("• <b>Limited dataset size:</b> ~50 images per class is insufficient to fully exploit a 25-million-parameter ResNet50, creating a performance ceiling.", S['bullet']),
        Paragraph("• <b>Accuracy ceiling:</b> 66.75% accuracy — fine-grained recognition across 80 visually similar classes on a small dataset is inherently challenging.", S['bullet']),
        Paragraph("• <b>Fine-grained confusion:</b> Visually similar classes (e.g., different dal preparations, rice dishes) lead to systematic misclassifications.", S['bullet']),
        Paragraph("• <b>Out-of-distribution generalization:</b> The model may perform poorly on images with very different lighting, camera angles, or plating styles.", S['bullet']),
        Paragraph("• <b>Static food information:</b> The food information dictionary covers only a subset of 80 classes; remaining classes show a placeholder message.", S['bullet']),
        Paragraph("• <b>Local deployment only:</b> The Streamlit application has not been deployed to a public server.", S['bullet']),
        Paragraph("• <b>No ingredient detection:</b> The system cannot visually identify individual ingredients; ingredient information is class-name mapped.", S['bullet']),
    ]

    story.append(Paragraph("7.3  Future Enhancements", S['sec_heading']))
    story += [
        Paragraph("• <b>Larger dataset:</b> Expanding to 200–500 images per class and adding more regional food categories would significantly improve accuracy.", S['bullet']),
        Paragraph("• <b>Vision Transformers (ViT):</b> Exploring transformer-based architectures that capture long-range spatial dependencies may outperform CNNs on fine-grained recognition.", S['bullet']),
        Paragraph("• <b>Attention mechanisms:</b> Integrating spatial attention or class activation mapping (CAM) to focus on discriminative food regions.", S['bullet']),
        Paragraph("• <b>Ensemble methods:</b> Combining predictions from multiple architectures (ResNet50, EfficientNet, ViT) for improved robustness.", S['bullet']),
        Paragraph("• <b>Mobile deployment:</b> Model quantization (INT8) and pruning to enable deployment on iOS/Android via ONNX or TFLite.", S['bullet']),
        Paragraph("• <b>Cloud deployment:</b> Hosting on AWS, GCP, or Streamlit Community Cloud for public access.", S['bullet']),
        Paragraph("• <b>Nutritional information:</b> Integrating a nutritional database API for calorie counts and macro-nutrient breakdowns.", S['bullet']),
        Paragraph("• <b>Multi-label classification:</b> Extending to identify multiple food items in one image (e.g., a thali plate).", S['bullet']),
        Paragraph("• <b>Active learning:</b> Allowing users to correct wrong predictions to continuously improve the model.", S['bullet']),
        PageBreak(),
    ]

    # ══════════════════════════════════════════════════════
    # REFERENCES
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("References", S['ch_heading']))
    story.append(HR())

    refs = [
        ("[1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image "
         "recognition. In <i>Proceedings of the IEEE Conference on Computer Vision and Pattern "
         "Recognition (CVPR)</i>, pp. 770–778."),
        ("[2] Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). "
         "ImageNet: A large-scale hierarchical image database. In <i>2009 IEEE Conference on "
         "Computer Vision and Pattern Recognition (CVPR)</i>, pp. 248–255."),
        ("[3] PyTorch Development Team. (n.d.). <i>PyTorch Documentation</i>. "
         "Retrieved from https://pytorch.org/docs/"),
        ("[4] Torchvision Contributors. (n.d.). <i>Torchvision Documentation</i>. "
         "Retrieved from https://pytorch.org/vision/stable/"),
        ("[5] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. "
         "<i>Journal of Machine Learning Research</i>, 12, 2825–2830."),
        ("[6] Streamlit Inc. (n.d.). <i>Streamlit Documentation</i>. "
         "Retrieved from https://docs.streamlit.io/"),
        ("[7] Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. "
         "In <i>International Conference on Learning Representations (ICLR)</i>."),
        ("[8] Szegedy, C., et al. (2016). Rethinking the inception architecture for computer "
         "vision. <i>CVPR</i>, pp. 2818–2826. (Label smoothing reference)"),
        ("[9] Zhong, Z., et al. (2020). Random erasing data augmentation. "
         "<i>AAAI Conference on Artificial Intelligence</i>, Vol. 34, pp. 13001–13008."),
    ]
    for ref in refs:
        story.append(Paragraph(ref, S['body']))
        story.append(SP(0.05))

    # ── Build ─────────────────────────────────────────────
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"\nReport saved -> {pdf_path}")


if __name__ == "__main__":
    build_report()
