# Retrospective PMOS Clinical Priority Triage Using LLMs
**Final Project for: Introduction to Language Processing with LLMs**


## 📖 Project Overview
This project develops a **Clinical Decision Support System (CDSS)** powered by Large Language Models (LLMs) to triage patient portal messages related to **Polyendocrine Metabolic Ovarian Syndrome (PMOS)** (formerly known as PCOS). 

In medical triage, NLP models often face a dangerous trade-off between **under-triage** (missing acute emergencies) and **alert fatigue** (overwhelming doctors with false positives). This project solves this by moving beyond simple text classification. We benchmarked multiple Zero-Shot LLMs, fine-tuned domain-specific transformers, and engineered a novel **Context-Aware Conversational Safety Cascade** to mathematically guarantee safe routing.

## 🧬 Data Generation & Augmentation
Since no off-the-shelf public dataset perfectly maps to the novel PMOS multi-system criteria, we generated a synthetic dataset of 3,000 longitudinal patient cases to solve this task. 
*   **Methodology:** We utilized an attribute-based text generation flow using GPT-4o. The prompt strategy combined structured clinical attributes (Age, Past Medical History) with varied linguistic profiles to simulate noisy, real-world patient portal messages (e.g., overly polite, vague, or anxious tones). 
*   This synthetic generation allowed us to explicitly control the class distribution and introduce challenging "Confidently Wrong" edge cases for the LLMs to evaluate.

## 🏥 Clinical Context: The Shift to PMOS
Based on recent 2024-2026 international medical consensus (The Lancet), *Polycystic Ovary Syndrome (PCOS)* is being formally renamed to *Polyendocrine Metabolic Ovarian Syndrome (PMOS)*. The new criteria emphasize multisystem metabolic and emotional symptoms (e.g., insulin resistance, cardiovascular risks, depression, anxiety). Our dataset and models are explicitly trained to recognize these complex, longitudinal symptom presentations rather than just isolated gynecological complaints.

## 🔬 Experimental Background & Models Tested
The core of this project relies on a systematic progression of NLP experiments. Clinical triage is highly nuanced; general-purpose LLMs often fail because they over-index on emotional language rather than medical severity. To prove this, we structured our experiments from basic benchmarking to advanced domain-specific architectures:

*   **`notebook1_final_dataset_preparation.ipynb`**: Data cleaning and longitudinal context fusion. We combined `Age + History + Current Message` to ensure models evaluate the patient's holistic medical background.
*   **`notebook2_final_EDA_baseline_model.ipynb`**: Exploratory Data Analysis and baseline traditional ML testing. Highlighted the extreme class imbalance inherent to medical data.
*   **`notebook3_6_models_zero_shot.ipynb`**: Tested general LLMs (Qwen, Mistral-7B, `facebook/bart-large-mnli`) using Zero-Shot Ordinal Regression. **Conclusion:** General models struggle heavily with clinical triage logic, suffering from the "Confidently Wrong" phenomenon.
*   **`notebook4_experiment_DistilBERT.ipynb`**: A lightweight transformer experiment utilizing `sklearn`'s `compute_class_weight` to force the model to focus on the minority (HIGH-risk) class.
*   **`notebook5_experiment_Bio_ClinicalBERT.ipynb`**: A model pre-trained natively on MIMIC-III clinical notes. Significantly outperformed general-domain models by understanding nuanced clinical terminology.
*   **`final_notebook_llm_evaluation_for_pmos.ipynb`**: The Master Architecture (`Bio_ClinicalBERT` + Generative Conversational AI). Implements a 99.5% Confidence Safety Net that mathematically fixes misclassifications before a final routing decision is made.
*   **`final_notebook_llm_evaluation_for_pmos_with_csv_of_results.ipynb`**: The Master Architecture (`Bio_ClinicalBERT` + Generative Conversational AI) re-run. Implements a 99.5% Confidence Safety Net that mathematically fixes misclassifications before a final routing decision is made.

---

## 📊 Model Performance Summary

### 1. Zero-Shot Models (Qwen, Mistral, BART)
*   **Accuracy Range:** ~0.34 to 0.42
*   **Observation:** Highly unreliable for clinical routing; frequently classified HIGH-risk emergencies as LOW/MEDIUM risk.

### 2. DistilBERT Fine-Tuned
*   **Accuracy:** 0.817

### 3. Bio_ClinicalBERT Fine-Tuned (Baseline)
*   **Accuracy:** 0.828 (Prior to continuous regression and safety extraction)

### 4. Final Master Architecture (Bio_ClinicalBERT + Conversational Safety Cascade)
*   **Final Accuracy:** 0.997 (99.7%)
*   **Recall for Emergencies:** 1.00 (Zero Dangerous Under-Triage).
*   **Misclassifications Fixed:** 29 dangerous baseline errors safely converted out of 251 pipeline interventions.
*   **Errors Introduced:** 0 (The safety layer broke zero correct predictions).
*   **Summary:** The addition of the Conversational Safety Cascade mathematically guaranteed emergency capture while autonomously clearing stable patients, mitigating alert fatigue.

## 🧠 NLP Methodology & Architecture
1.  **Contextual Enrichment:** Fused structured EHR data into the text string: `Age: X | History: Y | Message: Z`.
2.  **Balanced Loss Functions:** Injected class weights into a Custom HuggingFace `Trainer` to heavily penalize False Negatives.
3.  **Continuous Regression Extraction:** Applied a Softmax layer to the logits to extract a continuous probability score (0.0 to 1.0) to identify uncertain predictions.
4.  **The Conversational Safety Cascade:** If confidence drops below 99.5%, or if deterministic red flags are detected, the system intercepts the message for automated, conversational patient verification.

## 📥 Input/Output Examples
**Example 1 (Routine/Stable - Cleared by Tier-1):**
*   **Input:** `Age: 24 | History: Asthma | Message: "I'm sorry to hear that I've broken out again! It's been really frustrating..."`
*   **Output:** `System Route: LOW Risk` (Final Protocol Assurance: 100.0%)

**Example 2 (Uncertainty Intercept - Fixed to MEDIUM):**
*   **Input:** `Age: 31 | History: None | Message: "I've been feeling unusually tired after eating. It's like my body just needs more time..."`
*   **System Action:** Initial Confidence was 99.5%. Triggered AI Persona.
*   **Output:** `System Route: MEDIUM Risk` (Final Protocol Assurance: 99.7%)

## 🚀 How to Run
1. **Clone the repository:**

   git clone [https://github.com/AlinaKapelovich/Retrospective-PMOS-Clinical-Priority-Triage-Using-LLMs.git](https://github.com/AlinaKapelovich/Retrospective-PMOS-Clinical-Priority-Triage-Using-LLMs.git)
   cd Retrospective-PMOS-Clinical-Priority-Triage-Using-LLMs



2. **Install dependencies:**

pip install -r requirements.txt




3. **Run the pipeline:** Open the notebooks in Jupyter or Google Colab. For the fine-tuning notebooks, **a GPU (e.g., T4 on Colab) is highly recommended**.

## 📂 Repository Structure

* `bibliography/`: Clinical references and research papers used for domain context.
* `final_dataset/`: The simulated PMOS dataset and metadata used for training and testing.
* `ipynb_files/`: Jupyter notebooks containing data generation, EDA, zero-shot testing, and model fine-tuning.
* `results/`: Evaluation metrics, confusion matrices, and experiment outputs.
* `visuals/`: Presentation file.
* `README.md`: Project overview, methodology, and results.
* `requirements.txt`: Python dependencies required to run the notebooks.


## 👥 Team Members

* Alina Kapelovich
* Victoria Golovitsky

## 📚 Bibliography & Clinical References

1. **Dokras, Anuja, Elisabeth Stener-Victorin, Bulent O. Yildiz, et al.** “Androgen Excess- Polycystic Ovary Syndrome Society: Position Statement on Depression, Anxiety, Quality of Life, and Eating Disorders in Polycystic Ovary Syndrome.” Fertility and Sterility 109, no. 5 (2018): 888–99. https://doi.org/10.1016/j.fertnstert.2018.01.038.
2. **Peña, Alexia S., Selma Feldman Witchel, Jacky Boivin, et al.** “International Evidence-Based Recommendations for Polycystic Ovary Syndrome in Adolescents.” BMC Medicine 23, no. 1 (2025): 151. https://doi.org/10.1186/s12916-025-03901-w.
3. **“Revised 2003 Consensus on Diagnostic Criteria and Long-Term Health Risks Related to Polycystic Ovary Syndrome.”** Fertility and Sterility 81, no. 1 (2004): 19–25. https://doi.org/10.1016/j.fertnstert.2003.10.004.
4. **Stener-Victorin, Elisabet, Helena Teede, Robert J. Norman, et al.** “Polycystic Ovary Syndrome.” Nature Reviews Disease Primers 10, no. 1 (2024): 27. https://doi.org/10.1038/s41572-024-00511-3.
5. **Teede, Helena J.** Recommendations From the 2023 International Evidence-Based Guideline for the Assessment and Management of Polycystic Ovary Syndrome*. n.d.
**Teede, Helena J., Mahnaz Bahri Khomami, Rachel Morman, et al. “Polyendocrine Metabolic Ovarian Syndrome, the New Name for Polycystic Ovary Syndrome: A Multistep Global Consensus Process.” The Lancet** 407, no. 10545 (2026): 2329–39. https://doi.org/10.1016/S0140-6736(26)00717-8.
6. **Teede, Helena J., Mahnaz Bahri Khomami, Rachel Morman, et al.** “Polyendocrine Metabolic Ovarian Syndrome, the New Name for Polycystic Ovary Syndrome: A Multistep Global Consensus Process.” The Lancet 407, no. 10545 (2026): 2329–39. https://doi.org/10.1016/S0140-6736(26)00717-8.

