# 🧠 Breast Cancer Classification Using Deep Learning (TensorFlow)

This project leverages a supervised deep learning approach using **TensorFlow and Keras** to develop a high-performance binary classification model for early-stage breast cancer detection. By analyzing clinical cell nucleus features extracted from digitized images, the model can accurately predict whether a tumor is **Malignant** or **Benign**.

The system is trained end-to-end using a densely connected neural network with optimized hyperparameters, modern activation functions, and robust validation techniques to ensure generalization on unseen data.

---

## 📌 Highlights

- ✅ Implemented a deep neural network using **TensorFlow** and **Keras**
- 📊 Achieved **>96% accuracy** on unseen test data
- ⚙️ Trained on the **Breast Cancer Wisconsin (Diagnostic)** dataset
- 🧪 Integrated model evaluation, performance visualization, and live prediction capability
- 🧠 Demonstrates the power of AI in medical diagnostics

---

## 📊 Dataset Overview

- Sourced from the [UCI Machine Learning Repository] ; dataset has been provided in the repo
- Contains 30 numerical features (e.g., radius, texture, smoothness, concavity)
- Target labels:
  - `0`: Malignant (cancerous)
  - `1`: Benign (non-cancerous)

---

## 🧠 Model Architecture

- **Input Layer:** 30 features  
- **Hidden Layers:** Multiple fully connected (`Dense`) layers with **ReLU** activation  
- **Regularization:** Dropout layers to reduce overfitting  
- **Output Layer:** 1 neuron with **sigmoid** activation  
- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Evaluation Metric:** Accuracy  

---

## 📈 Training Progress

<img src="https://raw.githubusercontent.com/nithun-rajan/Breast-Cancer-classification-with-Neural-Network/main/Figure_1.png" width="700"/>

The model demonstrated steady convergence across 10 epochs, reaching over **97% training accuracy** and **~95.6% validation accuracy**, reflecting strong generalization.

---

## 📊 Performance Summary

| Metric               | Value     |
|----------------------|-----------|
| Training Accuracy    | 97.25%    |
| Validation Accuracy  | 95.65%    |
| Test Accuracy        | **96.49%** |
| Final Validation Loss| 0.1653    |
| Prediction Examples  | Malignant / Benign |

These results confirm that deep learning can be an effective tool for building diagnostic decision-support systems in oncology.

---

## 🔍 Inference Results

- **Case 1:** The person is likely to have **Malignant breast cancer**  
- **Case 2:** The person is likely to have **Benign breast cancer**

These outputs reflect the model’s real-time diagnostic capability based on unseen input vectors.

---

## 💻 How to Run Locally

```bash
git clone https://github.com/nithun-rajan/Breast-Cancer-classification-with-Neural-Network
cd Breast-Cancer-classification-with-Neural-Network
pip install -r requirements.txt
python main.py
