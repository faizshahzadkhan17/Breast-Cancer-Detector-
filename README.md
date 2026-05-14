# 🩺 Breast Cancer Detector — Full-Stack AI Application

> AI-powered breast cancer diagnosis from cell nucleus measurements using machine learning.

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://react.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

---

## 📖 Overview

This project wraps a pre-trained **Logistic Regression** model (97%+ accuracy) around the UCI Wisconsin Breast Cancer dataset into a modern full-stack web application. Users can:

- **Single Prediction** — Enter 30 cell-nucleus features for instant diagnosis
- **Batch Prediction** — Upload a CSV file for multiple patients at once
- **History** — View all past prediction records (Supabase)
- **Dashboard** — Summary stats, server status, and quick actions

### ML Pipeline

| Component | Details |
|-----------|---------|
| Dataset | UCI Wisconsin Diagnostic Breast Cancer (WDBC) |
| Features | 30 cell-nucleus measurements (mean, SE, worst) |
| Models trained | Logistic Regression, KNN, Random Forest, SVM, Decision Tree, Neural Network |
| Best model | Logistic Regression (~97% accuracy) |
| Scaler | StandardScaler (fitted on training split) |
| Labels | 0 = Benign, 1 = Malignant |

---

## 🏗️ Project Structure

```
Breast-Cancer-Detector-App/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── requirements.txt     # Backend Python deps
│   └── render.yaml          # Render.com deployment config
├── frontend/
│   ├── src/
│   │   ├── components/      # Reusable UI components
│   │   ├── pages/           # Dashboard, SinglePredict, BatchPredict, History
│   │   ├── api.js           # API utility
│   │   ├── App.jsx          # Root layout
│   │   ├── main.jsx         # Entry point with router
│   │   └── index.css        # Design system (Tailwind)
│   ├── index.html
│   ├── tailwind.config.js
│   └── vite.config.js
├── src/                     # Original ML pipeline
│   ├── model.py             # Training script
│   ├── best_model.joblib    # Pre-trained model
│   ├── scaler.joblib        # Fitted StandardScaler
│   └── predict_csv.py       # CLI batch predictor
├── data/                    # Dataset files
├── .env.example             # Environment variable template
├── requirements.txt         # Root Python dependencies
├── setup.bat                # Windows one-click setup
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Git**

### Option A: One-click Setup (Windows)

```bash
git clone https://github.com/Breast-Cancer-Detector/Breast-Cancer-Detector-.git
cd Breast-Cancer-Detector-
setup.bat
```

### Option B: Manual Setup

#### 1. Clone & Install Backend

```bash
git clone https://github.com/Breast-Cancer-Detector/Breast-Cancer-Detector-.git
cd Breast-Cancer-Detector-

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r backend/requirements.txt
```

#### 2. Configure Environment

```bash
copy .env.example .env
# Edit .env with your Supabase credentials (optional)
```

#### 3. Start Backend

```bash
cd backend
uvicorn main:app --reload
# Server runs at http://localhost:8000
```

#### 4. Install & Start Frontend

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
# App runs at http://localhost:5173
```

---

## 🗄️ Supabase Setup (Optional)

History features require a Supabase database. If you don't need history, the app works fine without it.

### 1. Create a Supabase Project
Go to [supabase.com](https://supabase.com), create a new project.

### 2. Run the SQL

In the Supabase SQL Editor, run:

```sql
CREATE TABLE predictions (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  patient_label TEXT,
  diagnosis TEXT NOT NULL,
  confidence FLOAT NOT NULL,
  input_features JSONB
);
```

### 3. Add Credentials to `.env`

```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-anon-key
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server health check |
| `GET` | `/features` | List of 30 expected feature names |
| `POST` | `/predict` | Single prediction (30 features → diagnosis) |
| `POST` | `/predict-batch` | Batch CSV prediction |
| `GET` | `/history` | Fetch all saved predictions |
| `POST` | `/history` | Save a prediction record |

---

## 🌐 Deployment

### Backend → Render.com

1. Push repo to GitHub
2. Connect to Render, select the repo
3. Use the `backend/render.yaml` blueprint
4. Add `SUPABASE_URL` and `SUPABASE_KEY` as environment variables

### Frontend → Vercel

1. Import frontend folder to Vercel
2. Set `VITE_API_URL` to your Render backend URL
3. Deploy

---

## 🧪 Environment Variables

| Variable | Where | Description |
|----------|-------|-------------|
| `SUPABASE_URL` | `.env` (root) | Supabase project URL |
| `SUPABASE_KEY` | `.env` (root) | Supabase anon/public key |
| `VITE_API_URL` | `frontend/.env` | Backend API URL (default: `http://localhost:8000`) |

---

## 📜 License

This project is for educational purposes. The original ML pipeline is from the [Breast-Cancer-Detector](https://github.com/Breast-Cancer-Detector/Breast-Cancer-Detector-) repository.
