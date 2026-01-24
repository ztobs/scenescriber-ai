#!/bin/bash

# SceneScriber AI Setup Script
# This script sets up both backend and frontend

echo "🎬 Setting up SceneScriber AI..."

# Check Python version
echo "Checking Python version..."
python3 --version

# Setup backend
echo -e "\n📦 Setting up Python backend..."
cd backend

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install minimal requirements
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements-minimal.txt

# Create necessary directories
echo "Creating upload and export directories..."
mkdir -p uploads exports

# Setup frontend
echo -e "\n⚛️  Setting up React frontend..."
cd ../frontend

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
npm install

echo -e "\n✅ Setup complete!"
echo -e "\nTo start the application:"
echo "1. Backend: cd backend && source venv/bin/activate && uvicorn src.main:app --reload --port 8000"
echo "2. Frontend: cd frontend && npm run dev"
echo -e "\nAccess points:"
echo "• Frontend: http://localhost:3000"
echo "• Backend API: http://localhost:8000"
echo "• API Docs: http://localhost:8000/docs"