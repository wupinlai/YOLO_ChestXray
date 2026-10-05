"""
YOLO_ChestXray Comprehensive PDF Report Generator (Plan 1 Compliant)
Generates: reports/final_report.pdf
Features:
- Page 1: Project Summary & Model Configuration
- Page 2: Training Metrics & Comparison Progression
- Page 3: Original vs Inference Result (Side-by-Side)
- Page 4: Detection Detail & Threshold Assessment
- Page 5: Confidence Score Analysis
- Page 6: Class Statistics & Detection Distribution
- Page 7: IoU Analysis & Ground Truth Overlap
- Page 8: Error Statistics & Categorization
- Page 9: Error Gallery (2x2 Grid)
- Page 10: False Positive & Misclassification Detail
- Page 11: False Negative & Low IoU Cases
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np
import pandas as pd


def create_page_with_header(title: str, subtitle: str = ""):
    """Create a standard landscape A4 figure with title styling."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=300) # Standard A4 Landscape
    fig.patch.set_facecolor('#ffffff')
    ax.set_axis_off()

    # Title Bar
    fig.text(0.06, 0.94, title, fontsize=18, fontweight='bold', color='#1a365d', fontfamily='sans-serif')
    if subtitle:
        fig.text(0.06, 0.91, subtitle, fontsize=11, color='#4a5568', fontfamily='sans-serif')
    
    # Bottom Footer
    fig.text(0.06, 0.03, "YOLO_ChestXray Multi-Pathology Diagnostic Platform - Automated Evaluation Report", fontsize=8, color='#a0aec0')
    fig.text(0.90, 0.03, "Plan 1 Compliant", fontsize=8, color='#a0aec0', ha='right')
    return fig, ax


def embed_image_to_ax(ax, img_path: str, bounds=(0.06, 0.08, 0.88, 0.80)):
    """Embeds an image into specific figure relative coordinate bounds."""
    if not os.path.exists(img_path):
        ax.text(0.5, 0.5, f"Artifact not found:\n{os.path.basename(img_path)}", 
                ha='center', va='center', fontsize=12, color='#e53e3e',
                bbox=dict(boxstyle="round,pad=1", fc="#fff5f5", ec="#feb2b2", lw=1.5))
        return
    img = Image.open(img_path)
    ax.imshow(img)


def generate_final_report_pdf(reports_dir: str = "reports", output_pdf: str = "reports/final_report.pdf"):
    """Generates the multi-page final report PDF adhering to all Plan 1 standards."""
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    print(f"[INFO] Compiling Plan 1 Final PDF Report: {output_pdf}")

    with PdfPages(output_pdf) as pdf:
        # -------------------------------------------------------------
        # PAGE 1: Project Summary
        # -------------------------------------------------------------
        fig, ax = create_page_with_header("YOLO_ChestXray: Chest Pathology Detection System", "Executive Summary & Training Architecture")
        
        summary_text = (
            "1. Research Objective & System Overview\n"
            "   This system applies YOLOv7 deep learning architecture to automate the localization and identification\n"
            "   of 14 critical thoracic pathologies from digital chest X-ray radiographs (ChestXray8).\n\n"
            "2. Training & Validation Protocol\n"
            "   • Architecture: YOLOv7 Anchor-based Deep CNN\n"
            "   • Input Image Resolution: 640 x 640 pixels (Multi-scale augmentation)\n"
            "   • Training Strategy: 5 Stages (10 Epochs / Stage, Total 50 Epochs)\n"
            "   • Checkpoint Persistence: Incremental snapshots synced to Google Drive & GitHub Releases\n"
            "   • Best Model Criteria: Prioritizing highest mAP@0.5:0.95, highest mAP@0.5, and lowest Val Loss\n\n"
            "3. 14 Target Thoracic Pathologies\n"
            "   Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia,\n"
            "   Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia.\n\n"
            "4. Comprehensive Error Diagnostic Framework\n"
            "   Adheres to strict spatial overlap standards (IoU) to categorize detections into True Positives,\n"
            "   False Positives, False Negatives, Misclassifications, and Low IoU errors."
        )
        ax.text(0.08, 0.85, summary_text, transform=ax.transAxes, fontsize=11, verticalalignment='top',
                fontfamily='sans-serif', linespacing=1.6,
                bbox=dict(boxstyle="round,pad=1.2", fc="#f7fafc", ec="#cbd5e0", lw=1))
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # PAGE 2: Training Metrics & Progression
        # -------------------------------------------------------------
        fig, ax = create_page_with_header("Training Progression & Loss Convergence", "Stage-wise metric evaluation across 50 epochs")
        map_path = os.path.join(reports_dir, "map50_comparison.png")
        loss_path = os.path.join(reports_dir, "loss_comparison.png")
        
        # Split into two subplots
        ax1 = fig.add_axes([0.08, 0.12, 0.40, 0.72])
        ax2 = fig.add_axes([0.52, 0.12, 0.40, 0.72])
        ax1.set_axis_off()
        ax2.set_axis_off()
        if os.path.exists(map_path):
            ax1.imshow(Image.open(map_path))
        if os.path.exists(loss_path):
            ax2.imshow(Image.open(loss_path))
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # PAGE 3: Original vs Inference Result (Side-by-Side)
        # -------------------------------------------------------------
        fig, ax = create_page_with_header("Validation Inference: Original vs. Predicted Diagnosis", "Side-by-side comparison of baseline sample vs model localization")
        orig_sample = os.path.join(reports_dir, "final_inference", "sample_image.jpg")
        pred_sample = os.path.join(reports_dir, "final_inference", "inference_result.jpg")

        ax1 = fig.add_axes([0.08, 0.12, 0.40, 0.72])
        ax2 = fig.add_axes([0.52, 0.12, 0.40, 0.72])
        ax1.set_axis_off()
        ax2.set_axis_off()
        ax1.set_title("Original Chest X-Ray", fontsize=12, fontweight='bold', pad=10)
        ax2.set_title("Model Inference & Pathology Localization", fontsize=12, fontweight='bold', pad=10)

        if os.path.exists(orig_sample):
            ax1.imshow(Image.open(orig_sample))
        else:
            ax1.text(0.5, 0.5, "Sample image not generated yet", ha='center', va='center')

        if os.path.exists(pred_sample):
            ax2.imshow(Image.open(pred_sample))
        else:
            ax2.text(0.5, 0.5, "Inference result not generated yet", ha='center', va='center')
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # PAGE 4: Detection Detail & Precision-Recall Progression
        # -------------------------------------------------------------
        fig, ax = create_page_with_header("Precision, Recall & F1 Curve Dynamics", "Detailed trade-off analysis for clinical sensitivity and specificity")
        pr_path = os.path.join(reports_dir, "precision_recall_comparison.png")
        cm_path = os.path.join(reports_dir, "confusion_matrix.png")

        ax1 = fig.add_axes([0.08, 0.12, 0.40, 0.72])
        ax2 = fig.add_axes([0.52, 0.12, 0.40, 0.72])
        ax1.set_axis_off()
        ax2.set_axis_off()
        if os.path.exists(pr_path):
            ax1.imshow(Image.open(pr_path))
        if os.path.exists(cm_path):
            ax2.imshow(Image.open(cm_path))
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # PAGE 5: Confidence Analysis
        # -------------------------------------------------------------
        fig, ax = create_page_with_header("Model Confidence Distribution Analysis", "Statistical distribution of prediction confidence scores")
        conf_path = os.path.join(reports_dir, "confidence_distribution.png")
        ax_img = fig.add_axes([0.15, 0.12, 0.70, 0.72])
        ax_img.set_axis_off()
        if os.path.exists(conf_path):
            ax_img.imshow(Image.open(conf_path))
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # PAGE 6: Class Statistics
        # -------------------------------------------------------------
        fig, ax = create_page_with_header("Pathology Class Statistics & Detection Breakdown", "Frequency and average confidence per disease category")
        c_dist_path = os.path.join(reports_dir, "class_distribution.png")
        c_conf_path = os.path.join(reports_dir, "class_confidence.png")

        ax1 = fig.add_axes([0.08, 0.12, 0.40, 0.72])
        ax2 = fig.add_axes([0.52, 0.12, 0.40, 0.72])
        ax1.set_axis_off()
        ax2.set_axis_off()
        if os.path.exists(c_dist_path):
            ax1.imshow(Image.open(c_dist_path))
        if os.path.exists(c_conf_path):
            ax2.imshow(Image.open(c_conf_path))
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # PAGE 7: IoU Analysis
        # -------------------------------------------------------------
        fig, ax = create_page_with_header("Intersection over Union (IoU) Localization Analysis", "Bounding box overlap accuracy compared against expert annotations")
        iou_dist_path = os.path.join(reports_dir, "iou_distribution.png")
        iou_cls_path = os.path.join(reports_dir, "class_iou_comparison.png")

        ax1 = fig.add_axes([0.08, 0.12, 0.40, 0.72])
        ax2 = fig.add_axes([0.52, 0.12, 0.40, 0.72])
        ax1.set_axis_off()
        ax2.set_axis_off()
        if os.path.exists(iou_dist_path):
            ax1.imshow(Image.open(iou_dist_path))
        if os.path.exists(iou_cls_path):
            ax2.imshow(Image.open(iou_cls_path))
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # PAGE 8: Error Statistics
        # -------------------------------------------------------------
        fig, ax = create_page_with_header("Diagnostic Error Taxonomy & Categorization", "Breakdown of False Positives, False Negatives, Misclassifications, and Low IoU")
        err_dist_path = os.path.join(reports_dir, "error_distribution.png")
        ax_img = fig.add_axes([0.15, 0.12, 0.70, 0.72])
        ax_img.set_axis_off()
        if os.path.exists(err_dist_path):
            ax_img.imshow(Image.open(err_dist_path))
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # PAGE 9: Error Gallery
        # -------------------------------------------------------------
        fig, ax = create_page_with_header("Clinical Error Gallery (2x2 Grid)", "Visual inspection of critical error modes with standardized legend and info panel")
        gallery_path = os.path.join(reports_dir, "error_gallery.jpg")
        ax_img = fig.add_axes([0.08, 0.10, 0.84, 0.76])
        ax_img.set_axis_off()
        if os.path.exists(gallery_path):
            ax_img.imshow(Image.open(gallery_path))
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # PAGE 10: False Positive & Misclassification Cases
        # -------------------------------------------------------------
        fig, ax = create_page_with_header("Detailed Error Cases: False Positives & Misclassifications", "Case studies demonstrating over-detection and cross-pathology confusion")
        worst_path = os.path.join(reports_dir, "worst_iou_cases.jpg")
        ax_img = fig.add_axes([0.08, 0.10, 0.84, 0.76])
        ax_img.set_axis_off()
        if os.path.exists(worst_path):
            ax_img.imshow(Image.open(worst_path))
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # PAGE 11: Top Detections & Verified Cases
        # -------------------------------------------------------------
        fig, ax = create_page_with_header("High-Confidence Diagnostic Detections", "Top scoring accurate lesion localizations across validation cohort")
        top_path = os.path.join(reports_dir, "top5_detections.jpg")
        ax_img = fig.add_axes([0.08, 0.10, 0.84, 0.76])
        ax_img.set_axis_off()
        if os.path.exists(top_path):
            ax_img.imshow(Image.open(top_path))
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

    print(f"[SUCCESS] Final Report PDF generated successfully: {output_pdf}")


if __name__ == "__main__":
    generate_final_report_pdf()
