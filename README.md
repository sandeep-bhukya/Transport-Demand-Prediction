# 🚌 Transport Demand Prediction - Full Project

A ready-to-run **Data Science Regression Project** that predicts the number of seats sold (transport demand).

## 📦 Contents
- `main.py` → trains the model, evaluates metrics, and shows graphs
- `app.py` → Streamlit web app for predictions
- `requirements.txt` → list of dependencies
- `best_model.pkl` → saved after running `main.py`
- `train_revised.csv` → dataset (add your CSV here)

## 🧠 How to Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Train & Visualize
```bash
python main.py
```

### 3️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

Then open the browser link (usually http://localhost:8501).

## ⚠️ Note for Windows Users
If you see this error:
```
ImportError: cannot import name 'builder' from 'google.protobuf.internal'
```
Run these commands:
```bash
pip uninstall protobuf -y
pip install protobuf==3.20.3
```

Or upgrade Streamlit:
```bash
pip install --upgrade streamlit
```

✅ After fixing, rerun the app:
```bash
streamlit run app.py
```