# 🚨 Anomaly Detector with Streamlit

An interactive anomaly detection app using **Isolation Forest** and **Streamlit**, enabling you to detect anomalies in your CSV data without coding.

---

## 📈 Features

✅ Upload your CSV dataset for anomaly detection  
✅ Uses **Isolation Forest** for unsupervised anomaly detection  
✅ Visualizes anomaly vs normal counts using Seaborn  
✅ Highlights detected anomalies in the table  
✅ Allows downloading results as CSV  
✅ **Deployed on Streamlit Cloud** for easy access

---

## 🚀 Live Demo

🔗 **[Try the app here](https://anomaly-detector-app.streamlit.app/)**

---

## ⚙️ Tech Stack

- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib
- Joblib

---

## 📂 How to Run Locally

1️⃣ Clone the repository:

```bash
git clone https://github.com/Manisha1808/anomaly-detector-streamlit.git
cd anomaly-detector-streamlit

2️⃣ Create and activate a virtual environment:

python -m venv venv
# Activate:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

3️⃣ Install dependencies:

pip install -r requirements.txt

4️⃣ Run the app:

streamlit run app.py

How It Works
Training (train.py):

Trains an Isolation Forest on your dataset and saves the model + scaler.

Inference (app.py):

Upload a CSV file.

Data is scaled and passed through the trained Isolation Forest.

Detects anomalies (-1) and marks them visually.

Allows downloading results.

💡 Future Improvements
Add support for different anomaly detection models.

Enable hyperparameter tuning from the UI.

Add authentication for sensitive data uploads.

Export charts as images for reports.

🙌 Acknowledgments
This project is part of Manisha's Data Science learning journey, building practical ML deployment skills for internships and placements.



