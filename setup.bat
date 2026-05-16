@echo off
echo ============================================
echo   Breast Cancer Detector - Setup Script
echo ============================================
echo.

REM --- Backend Setup ---
echo [1/5] Creating Python virtual environment...
python -m venv venv
call venv\Scripts\activate

echo [2/5] Installing Python dependencies...
pip install -r requirements.txt
pip install -r backend\requirements.txt

echo [3/5] Creating .env from template...
if not exist .env (
    copy .env.example .env
    echo       Created .env — please fill in your Supabase credentials!
) else (
    echo       .env already exists, skipping.
)

REM --- Frontend Setup ---
echo [4/5] Installing frontend dependencies...
cd frontend
call npm install
if not exist .env (
    copy .env.example .env
    echo       Created frontend .env
) else (
    echo       frontend .env already exists, skipping.
)
cd ..

echo.
echo [5/5] Setup complete!
echo.
echo  Next steps:
echo    1. Edit .env with your Supabase URL and Key
echo    2. Start backend:  cd backend ^& uvicorn main:app --reload
echo    3. Start frontend: cd frontend ^& npm run dev
echo.
pause
