# LLM_evaluation_for_PMOS
Final project for course HIT Introduction to LLM

# Retrospective PMOS Clinical Priority Triage Using LLMs
# PMOS Clinical Triage: Multi-Stage LLM Safety Architecture

**Final Project for: Introduction to Language Processing with LLMs**

## 📖 Project Overview
This project develops a **Clinical Decision Support System (CDSS)** powered by Large Language Models (LLMs) to triage patient portal messages related to **Polyendocrine Metabolic Ovarian Syndrome (PMOS)** (formerly known as PCOS). 

In medical triage, NLP models often face a dangerous trade-off between **under-triage** (missing acute emergencies) and **alert fatigue** (overwhelming doctors with false positives). This project solves this by moving beyond simple text classification. We benchmarked multiple Zero-Shot LLMs, fine-tuned domain-specific transformers, and engineered a novel **Context-Aware Conversational Safety Cascade** to mathematically guarantee safe routing.

## Clinical Context: The Shift to PMOS
Based on recent 2024-2026 international medical consensus (The Lancet), *Polycystic Ovary Syndrome (PCOS)* is being formally renamed to *Polyendocrine Metabolic Ovarian Syndrome (PMOS)*. The new criteria emphasize multisystem metabolic and emotional symptoms (e.g., insulin resistance, cardiovascular risks, depression, anxiety). Our dataset and models are explicitly trained to recognize these complex, longitudinal symptom presentations rather than just isolated gynecological complaints.

## Experimental Background & Models Tested

The core of this project relies on a systematic progression of NLP experiments. Clinical triage is highly nuanced; general-purpose LLMs often fail because they over-index on emotional language rather than medical severity. To prove this, we structured our experiments from basic benchmarking to advanced domain-specific architectures:

* **`notebook1_final_dataset_preparation.ipynb`**: 
  * **Goal:** Data cleaning and longitudinal context fusion. We combined `Age + History + Current Message` to ensure models evaluate the patient's holistic medical background, not just their current complaint.
* **`notebook2_final_EDA_baseline_model.ipynb`**: 
  * **Goal:** Exploratory Data Analysis and baseline traditional ML testing. Highlighted the extreme class imbalance inherent to medical data (many LOW risk, very few HIGH risk patients).
* **`notebook3_6_models_zero_shot.ipynb`**: 
  * **Models Tested:** Qwen, Mistral-7B-OpenOrca, and `facebook/bart-large-mnli`.
  * **Background:** We utilized Zero-Shot prompting and Ordinal Regression to establish a baseline. **Conclusion:** General models struggle heavily with clinical triage logic, suffering from the "Confidently Wrong" phenomenon. Fine-tuning was strictly required.
* **`notebook4_experiment_DistilBERT.ipynb`**: 
  * **Model Tested:** `DistilBERT`.
  * **Background:** A lightweight, computationally efficient transformer experiment. We applied `sklearn`'s `compute_class_weight` to force the model to focus on the minority (HIGH-risk) class, proving that size isn't everything if the loss function is properly weighted.
* **`notebook5_experiment_Bio_ClinicalBERT.ipynb`**: 
  * **Model Tested:** `emilyalsentzer/Bio_ClinicalBERT`.
  * **Background:** A model pre-trained natively on MIMIC-III clinical notes. This domain-specific transformer significantly outperformed the general-domain models by successfully understanding nuanced clinical terminology and historical EHR data.
* **`final_notebook_llm_evaluation_for_pmos.ipynb`**: 
  * **The Master Architecture:** `Bio_ClinicalBERT` + Generative Conversational AI.
  * **Background:** Implements a 99.5% Confidence Safety Net. If the baseline model is uncertain, a Context-Aware AI Persona is deployed to ask targeted follow-up questions, mathematically fixing misclassifications before a final routing decision is made.

---

## 📊 Model Performance Summary

### 1. Qwen Zero-Shot Classification
* **Accuracy:** 0.342
* **Confusion Matrix:**

             Pred LOW  Pred MED  Pred HIGH
  True LOW          7       137         96
  True MED         13       173         84
  True HIGH         0        65         25

### 2. Mistral-7B-OpenOrca Zero-Shot Classification

* **Accuracy:** 0.423
* **Confusion Matrix:**


             Pred LOW  Pred MED  Pred HIGH
  True LOW          0         0        240
  True MED          0       252         18
  True HIGH         0        86          4

### 3. DistilBERT Fine-Tuned

* **Accuracy:** 0.817
* **Confusion Matrix:**


             Pred LOW  Pred MED  Pred HIGH
  True LOW        209        31          0
  True MED         18       233         19
  True HIGH         0        36         54

### 4. Bio_ClinicalBERT Fine-Tuned

* **Accuracy:** 0.828
* **Confusion Matrix:**

             Pred LOW  Pred MED  Pred HIGH
  True LOW        209        30          1
  True MED         14       234         22
  True HIGH         1        28         61

### 5. Mistral-7B QLoRA Fine-Tuned

* **Accuracy:** 0.440
* **Confusion Matrix:**

             Pred LOW  Pred MED  Pred HIGH
  True LOW          7       137         96
  True MED          0       252         18
  True HIGH         0        86          4

### 6. Zero-Shot Ordinal Regression (facebook/bart-large-mnli)

* **Accuracy:** 0.413
* **Classification Report:**

               precision    recall  f1-score   support

          LOW       0.40      0.84      0.54       240
       MEDIUM       0.49      0.17      0.26       270
         HIGH       0.00      0.00      0.00        90

     accuracy                           0.41       600
    macro avg       0.30      0.34      0.27       600
 weighted avg       0.38      0.41      0.33       600

### 7. Final Master Architecture (Bio_ClinicalBERT + Conversational Safety Cascade)
Recall for Emergencies: 1.00 (Zero Dangerous Under-Triage).
Misclassifications Fixed: 29 dangerous baseline errors safely converted.
Errors Introduced: 0 (The safety layer broke zero correct predictions).
Summary: The addition of the Conversational Safety Cascade mathematically guaranteed emergency capture while autonomously clearing stable patients, mitigating alert fatigue.
---

## 🧠 NLP Methodology & Architecture

1. **Contextual Enrichment:** Fused structured EHR data into the text string: `Age: X | History: Y | Message: Z`.
2. **Balanced Loss Functions:** Injected class weights into a Custom HuggingFace `Trainer` to heavily penalize False Negatives.
3. **Continuous Regression Extraction:** Applied a Softmax layer to the logits to extract a continuous probability score (0.0 to 1.0) to identify uncertain predictions.
4. **The Conversational Safety Cascade:** If confidence drops below 99.5%, or if deterministic red flags are detected, the system intercepts the message for automated, conversational patient verification.

## 🏆 Key Results

The final cascaded architecture achieved state-of-the-art safety metrics on our test set:

* **Zero Dangerous Under-Triage:** Achieved a perfect **1.00 Recall** for HIGH and MEDIUM risk patients.
* **Flawless Conversion:** The Conversational Safety Protocol fixed **29** dangerous baseline misclassifications without breaking a single correct prediction (**0 errors introduced**).
* **Alert Fatigue Mitigation:** The AI successfully cleared stable patients autonomously, proving that safety does not have to overwhelm hospital staff.

## 🚀 How to Run

**1. Clone the repository:**

```bash
git clone [https://github.com/AlinaKapelovich/pmos-llm-triage.git](https://github.com/AlinaKapelovich/pmos-llm-triage.git)
cd pmos-llm-triage

```

**2. Install dependencies:**

```bash
pip install -r requirements.txt

```

**3. Run the pipeline:**
Open the notebooks in Jupyter or Google Colab. For the fine-tuning notebooks (4, 5, and final), **a GPU (e.g., T4 on Colab) is highly recommended** to accelerate processing.

---

## 📚 Bibliography & Clinical References

The architectural rules, keyword triggers, and contextual feature engineering in this project were deeply informed by the following evidence-based medical literature regarding PMOS:

1. **Dokras, A., Stener-Victorin, E., Yildiz, B. O., et al.** (2018). "Androgen Excess- Polycystic Ovary Syndrome Society: Position Statement on Depression, Anxiety, Quality of Life, and Eating Disorders in Polycystic Ovary Syndrome." *Fertility and Sterility*, 109(5), 888–899. https://doi.org/10.1016/j.fertnstert.2018.01.038
2. **Peña, A. S., Witchel, S. F., Boivin, J., et al.** (2025). "International Evidence-Based Recommendations for Polycystic Ovary Syndrome in Adolescents." *BMC Medicine*, 23(1), 151. https://doi.org/10.1186/s12916-025-03901-w
3. **Rotterdam ESHRE/ASRM-Sponsored PCOS Consensus Workshop Group.** (2004). "Revised 2003 Consensus on Diagnostic Criteria and Long-Term Health Risks Related to Polycystic Ovary Syndrome." *Fertility and Sterility*, 81(1), 19–25. https://doi.org/10.1016/j.fertnstert.2003.10.004
4. **Stener-Victorin, E., Teede, H., Norman, R. J., et al.** (2024). "Polycystic Ovary Syndrome." *Nature Reviews Disease Primers*, 10(1), 27. https://doi.org/10.1038/s41572-024-00511-3
5. **Teede, H. J.** (n.d.). *Recommendations From the 2023 International Evidence-Based Guideline for the Assessment and Management of Polycystic Ovary Syndrome*.
6. **Teede, H. J., Bahri Khomami, M., Morman, R., et al.** (2026). "Polyendocrine Metabolic Ovarian Syndrome, the New Name for Polycystic Ovary Syndrome: A Multistep Global Consensus Process." *The Lancet*, 407(10545), 2329–2339. https://doi.org/10.1016/S0140-6736(26)00717-8
