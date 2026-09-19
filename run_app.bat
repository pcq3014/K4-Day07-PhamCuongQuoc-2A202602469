@echo off
title UniRAG Web UI - K4-L3A
echo ========================================================
echo  UniRAG - He thong RAG Tra cuu Quy che & Dich vu Dai hoc
echo  Bien the K4-L3A - Lab Day 7
echo ========================================================
echo.

if exist .venv\Scripts\python.exe (
    echo [1/2] Kich hoat moi truong ao .venv...
    call .venv\Scripts\activate.bat
    echo [2/2] Khoi dong may chu Web UI tai http://localhost:8000 ...
    start http://localhost:8000
    python server.py 8000
) else (
    echo [!] Khong tim thay .venv, su dung python he thong...
    start http://localhost:8000
    python server.py 8000
)

pause
