#!/bin/bash

# Quick Start Script for SceneScriber AI
# Simple one-command startup for both services

set -e

echo "🚀 Starting SceneScriber AI..."

# Check if setup was done
if [ ! -d "backend/venv" ] || [ ! -d "frontend/node_modules" ]; then
    echo "❌ Setup not complete. Please run ./setup.sh first."
    exit 1
fi

# Function to handle cleanup
cleanup() {
    echo "🛑 Stopping services..."
    pkill -f "uvicorn src.main:app" 2>/dev/null || true
    pkill -f "npm run dev" 2>/dev/null || true
    echo "✅ Services stopped."
}

trap cleanup EXIT

# Start backend
echo "🔧 Starting backend on port 8000..."
cd backend
source venv/bin/activate
mkdir -p uploads exports
uvicorn src.main:app --reload --port 8000 --host 0.0.0.0 > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Start frontend
echo "🎨 Starting frontend on port 3000..."
cd frontend
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait a moment for services to start
sleep 5

echo ""
echo "✅ SceneScriber AI is running!"
echo ""
echo "🌐 Access:"
echo "   • Frontend: http://localhost:3000"
echo "   • Backend API: http://localhost:8000"
echo "   • API Docs: http://localhost:8000/docs"
echo ""
echo "📝 Logs:"
echo "   • Backend: tail -f backend.log"
echo "   • Frontend: tail -f frontend.log"
echo ""
echo "🛑 Press Ctrl+C to stop all services"
echo ""

# Keep running
wait $BACKEND_PID $FRONTEND_PID