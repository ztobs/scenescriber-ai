# SceneScriber AI Makefile
# Comprehensive management for Video Scene AI Analyzer

.PHONY: help setup start stop force-stop restart clean logs status test vram-cleanup install-llava update

# Default target
.DEFAULT_GOAL := help

# Project directories
BACKEND_DIR := backend
FRONTEND_DIR := frontend
VENV_DIR := $(BACKEND_DIR)/venv
VENV_ACTIVATE := $(VENV_DIR)/bin/activate

# Service ports
BACKEND_PORT := 8000
FRONTEND_PORT := 3000

# Log files
BACKEND_LOG := backend.log
FRONTEND_LOG := frontend.log

# PID files for service management
BACKEND_PID_FILE := .backend.pid
FRONTEND_PID_FILE := .frontend.pid

# Help documentation
help:
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║                 🎬 SceneScriber AI                       ║"
	@echo "║           Video Scene Analyzer Management                ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📦 Setup & Installation:"
	@echo "  make setup              - Full project setup (backend + frontend)"
	@echo "  make install-llava      - Install LLaVA dependencies for local AI"
	@echo "  make update             - Update all dependencies"
	@echo ""
	@echo "🚀 Service Management:"
	@echo "  make start              - Start both backend and frontend"
	@echo "  make start-backend      - Start only backend service"
	@echo "  make start-frontend     - Start only frontend service"
	@echo "  make stop               - Stop all services (graceful)"
	@echo "  make force-stop         - Force stop all services"
	@echo "  make restart            - Restart all services"
	@echo "  make status             - Check service status"
	@echo ""
	@echo "🔧 Development & Testing:"
	@echo "  make test               - Run all tests"
	@echo "  make test-backend       - Run backend tests"
	@echo "  make test-frontend      - Run frontend tests"
	@echo "  make lint               - Run linting on both backend and frontend"
	@echo "  make format             - Format code"
	@echo ""
	@echo "🧹 Cleanup & Maintenance:"
	@echo "  make clean              - Clean temporary files and logs"
	@echo "  make clean-all          - Full cleanup (including node_modules)"
	@echo "  make vram-cleanup       - Clear GPU VRAM (for LLaVA users)"
	@echo "  make logs               - Show service logs"
	@echo ""
	@echo "📊 System Information:"
	@echo "  make info               - Show system and project info"
	@echo "  make ports              - Check used ports"
	@echo "  make deps               - Show dependency versions"
	@echo ""
	@echo "🌐 Access URLs:"
	@echo "  Frontend:    http://localhost:3000"
	@echo "  Backend API: http://localhost:8000"
	@echo "  API Docs:    http://localhost:8000/docs"
	@echo ""
	@echo "💡 Tips:"
	@echo "  • Use 'make help' to show this message"
	@echo "  • Services run in background, use 'make logs' to view output"
	@echo "  • Use 'make vram-cleanup' if LLaVA runs out of GPU memory"
	@echo ""

# Setup targets
setup: check-prerequisites setup-backend setup-frontend
	@echo "✅ Setup complete!"
	@echo "Run 'make start' to launch the application"

check-prerequisites:
	@echo "🔍 Checking prerequisites..."
	@command -v python3 >/dev/null 2>&1 || { echo "❌ Python3 not found"; exit 1; }
	@command -v node >/dev/null 2>&1 || { echo "❌ Node.js not found"; exit 1; }
	@command -v npm >/dev/null 2>&1 || { echo "❌ npm not found"; exit 1; }
	@echo "✅ All prerequisites found"

setup-backend:
	@echo "📦 Setting up Python backend..."
	@cd $(BACKEND_DIR) && \
	if [ ! -d "venv" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv venv; \
	fi
	@. $(VENV_ACTIVATE) && \
	cd $(BACKEND_DIR) && \
	pip install --upgrade pip && \
	pip install -r requirements-minimal.txt
	@mkdir -p $(BACKEND_DIR)/uploads $(BACKEND_DIR)/exports
	@if [ ! -f "$(BACKEND_DIR)/.env" ]; then \
		cp $(BACKEND_DIR)/.env.example $(BACKEND_DIR)/.env; \
		echo "⚠️  Created .env file. Edit it to add your API keys."; \
	fi
	@echo "✅ Backend setup complete"

setup-frontend:
	@echo "⚛️  Setting up React frontend..."
	@cd $(FRONTEND_DIR) && npm install
	@echo "✅ Frontend setup complete"

install-llava:
	@echo "🤖 Installing LLaVA dependencies..."
	@. $(VENV_ACTIVATE) && \
	cd $(BACKEND_DIR) && \
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
	pip install transformers>=4.36.0 pillow bitsandbytes accelerate
	@echo "✅ LLaVA dependencies installed"
	@echo "💡 Note: LLaVA requires ~8GB VRAM. Use 'make vram-cleanup' if needed."

update:
	@echo "🔄 Updating dependencies..."
	@. $(VENV_ACTIVATE) && \
	cd $(BACKEND_DIR) && \
	pip install --upgrade -r requirements-minimal.txt
	@cd $(FRONTEND_DIR) && npm update
	@echo "✅ Dependencies updated"

# Service management targets
start: stop start-backend start-frontend status
	@echo "✅ Both services started!"
	@echo "🌐 Access URLs:"
	@echo "  Frontend:    http://localhost:$(FRONTEND_PORT)"
	@echo "  Backend API: http://localhost:$(BACKEND_PORT)"
	@echo "  API Docs:    http://localhost:$(BACKEND_PORT)/docs"
	@echo "📝 View logs: make logs"

start-backend:
	@echo "🔧 Starting backend on port $(BACKEND_PORT)..."
	@if [ ! -f "$(VENV_ACTIVATE)" ]; then \
		echo "❌ Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@. $(VENV_ACTIVATE) && \
	cd $(BACKEND_DIR) && \
	(uvicorn src.main:app --reload --port $(BACKEND_PORT) --host 0.0.0.0 > ../$(BACKEND_LOG) 2>&1 & echo $$! > ../$(BACKEND_PID_FILE))
	@sleep 3
	@if curl -s -f http://localhost:$(BACKEND_PORT)/ > /dev/null 2>&1; then \
		echo "✅ Backend is running"; \
	else \
		echo "❌ Backend failed to start. Check $(BACKEND_LOG)"; \
		exit 1; \
	fi

start-frontend:
	@echo "🎨 Starting frontend on port $(FRONTEND_PORT)..."
	@if [ ! -d "$(FRONTEND_DIR)/node_modules" ]; then \
		echo "❌ node_modules not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@cd $(FRONTEND_DIR) && \
	(npm run dev > ../$(FRONTEND_LOG) 2>&1 & echo $$! > ../$(FRONTEND_PID_FILE))
	@sleep 5
	@if curl -s http://localhost:$(FRONTEND_PORT) > /dev/null; then \
		echo "✅ Frontend is running"; \
	else \
		echo "⚠️  Frontend may still be starting..."; \
		sleep 5; \
		if curl -s http://localhost:$(FRONTEND_PORT) > /dev/null; then \
			echo "✅ Frontend is now running"; \
		else \
			echo "❌ Frontend failed to start. Check $(FRONTEND_LOG)"; \
			exit 1; \
		fi \
	fi

stop:
	@echo "🛑 Stopping services..."
	@if [ -f "$(BACKEND_PID_FILE)" ]; then \
		if kill -0 $$(cat $(BACKEND_PID_FILE)) 2>/dev/null; then \
			kill $$(cat $(BACKEND_PID_FILE)) 2>/dev/null || true; \
			echo "✅ Backend stopped (PID: $$(cat $(BACKEND_PID_FILE)))"; \
		else \
			echo "⚠️  Backend PID $$(cat $(BACKEND_PID_FILE)) not running"; \
		fi; \
		rm -f $(BACKEND_PID_FILE); \
	else \
		echo "⚠️  Backend not running (no PID file)"; \
	fi
	@if [ -f "$(FRONTEND_PID_FILE)" ]; then \
		if kill -0 $$(cat $(FRONTEND_PID_FILE)) 2>/dev/null; then \
			kill $$(cat $(FRONTEND_PID_FILE)) 2>/dev/null || true; \
			echo "✅ Frontend stopped (PID: $$(cat $(FRONTEND_PID_FILE)))"; \
		else \
			echo "⚠️  Frontend PID $$(cat $(FRONTEND_PID_FILE)) not running"; \
		fi; \
		rm -f $(FRONTEND_PID_FILE); \
	else \
		echo "⚠️  Frontend not running (no PID file)"; \
	fi
	@# Ensure frontend is killed by checking port 3000
	@if lsof -t -i:$(FRONTEND_PORT) >/dev/null 2>&1; then \
		echo "🧹 Cleaning up lingering frontend process on port $(FRONTEND_PORT)..."; \
		kill -9 $$(lsof -t -i:$(FRONTEND_PORT)) 2>/dev/null || true; \
		echo "✅ Frontend force killed"; \
	fi
	@# Ensure backend is killed by checking port 8000
	@if lsof -t -i:$(BACKEND_PORT) >/dev/null 2>&1; then \
		echo "🧹 Cleaning up lingering backend process on port $(BACKEND_PORT)..."; \
		kill -9 $$(lsof -t -i:$(BACKEND_PORT)) 2>/dev/null || true; \
		echo "✅ Backend force killed"; \
	fi

force-stop:
	@echo "🛑 Force stopping all services..."
	@pkill -f "uvicorn src.main:app" 2>/dev/null || true
	@pkill -f "npm run dev" 2>/dev/null || true
	@rm -f $(BACKEND_PID_FILE) $(FRONTEND_PID_FILE)
	@echo "✅ All services force stopped"

restart: stop start
	@echo "✅ Services restarted"

status:
	@echo "📊 Service Status:"
	@if [ -f "$(BACKEND_PID_FILE)" ] && kill -0 $$(cat $(BACKEND_PID_FILE)) 2>/dev/null; then \
		echo "  Backend:  RUNNING (PID: $$(cat $(BACKEND_PID_FILE)))"; \
	else \
		echo "  Backend:  STOPPED"; \
	fi
	@if [ -f "$(FRONTEND_PID_FILE)" ] && kill -0 $$(cat $(FRONTEND_PID_FILE)) 2>/dev/null; then \
		echo "  Frontend: RUNNING (PID: $$(cat $(FRONTEND_PID_FILE)))"; \
	else \
		echo "  Frontend: STOPPED"; \
	fi
	@echo ""
	@echo "🌐 Port Status:"
	@if lsof -i:$(BACKEND_PORT) >/dev/null 2>&1; then \
		echo "  Port $(BACKEND_PORT): IN USE (Backend)"; \
	else \
		echo "  Port $(BACKEND_PORT): AVAILABLE"; \
	fi
	@if lsof -i:$(FRONTEND_PORT) >/dev/null 2>&1; then \
		echo "  Port $(FRONTEND_PORT): IN USE (Frontend)"; \
	else \
		echo "  Port $(FRONTEND_PORT): AVAILABLE"; \
	fi

# Development & testing targets
test: test-backend test-frontend
	@echo "✅ All tests passed!"

test-backend:
	@echo "🧪 Running backend tests..."
	@. $(VENV_ACTIVATE) && \
	cd $(BACKEND_DIR) && \
	python -m pytest tests/ -v

test-frontend:
	@echo "🧪 Running frontend tests..."
	@cd $(FRONTEND_DIR) && \
	npm test -- --watchAll=false

lint:
	@echo "🔍 Linting code..."
	@. $(VENV_ACTIVATE) && \
	cd $(BACKEND_DIR) && \
	python -m black --check src/ && \
	python -m flake8 src/
	@cd $(FRONTEND_DIR) && \
	npm run lint

format:
	@echo "🎨 Formatting code..."
	@. $(VENV_ACTIVATE) && \
	cd $(BACKEND_DIR) && \
	python -m black src/ && \
	python -m isort src/
	@cd $(FRONTEND_DIR) && \
	npm run format

# Cleanup targets
clean:
	@echo "🧹 Cleaning temporary files..."
	@rm -f $(BACKEND_LOG) $(FRONTEND_LOG)
	@rm -f $(BACKEND_PID_FILE) $(FRONTEND_PID_FILE)
	@rm -rf $(BACKEND_DIR)/__pycache__ $(BACKEND_DIR)/tests/__pycache__
	@rm -rf $(BACKEND_DIR)/uploads/* $(BACKEND_DIR)/exports/*
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete"

clean-all: clean
	@echo "🧹 Deep cleaning..."
	@rm -rf $(VENV_DIR)
	@cd $(FRONTEND_DIR) && rm -rf node_modules
	@echo "✅ Full cleanup complete"
	@echo "⚠️  Run 'make setup' to reinstall dependencies"

vram-cleanup:
	@echo "🧠 Cleaning GPU VRAM..."
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "GPU Memory Before:"; \
		nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv; \
		echo ""; \
		echo "Clearing PyTorch CUDA cache..."; \
		. $(VENV_ACTIVATE) && \
		python -c "import torch; torch.cuda.empty_cache(); print('CUDA cache cleared')" 2>/dev/null || echo "CUDA not available"; \
		echo ""; \
		echo "GPU Memory After:"; \
		nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv; \
	else \
		echo "⚠️  nvidia-smi not found. Trying Python cleanup..."; \
		. $(VENV_ACTIVATE) && \
		python -c "import torch; torch.cuda.empty_cache(); print('CUDA cache cleared')" 2>/dev/null || echo "CUDA not available or not installed"; \
	fi
	@echo "✅ VRAM cleanup attempted"

logs:
	@echo "📝 Service Logs:"
	@echo "Backend Log ($(BACKEND_LOG)):"
	@if [ -f "$(BACKEND_LOG)" ]; then \
		tail -20 $(BACKEND_LOG); \
	else \
		echo "No backend log file found"; \
	fi
	@echo ""
	@echo "Frontend Log ($(FRONTEND_LOG)):"
	@if [ -f "$(FRONTEND_LOG)" ]; then \
		tail -20 $(FRONTEND_LOG); \
	else \
		echo "No frontend log file found"; \
	fi
	@echo ""
	@echo "💡 Tip: Use 'tail -f $(BACKEND_LOG)' to follow logs"

# System information targets
info:
	@echo "📊 System Information:"
	@echo "  Python: $$(python3 --version 2>/dev/null || echo 'Not found')"
	@echo "  Node.js: $$(node --version 2>/dev/null || echo 'Not found')"
	@echo "  npm: $$(npm --version 2>/dev/null || echo 'Not found')"
	@echo ""
	@echo "🎬 SceneScriber AI:"
	@echo "  Backend: $$(cd $(BACKEND_DIR) && . $(VENV_ACTIVATE) && python -c "import sys; print(f'Python {sys.version}')" 2>/dev/null || echo 'Not setup')"
	@echo "  Frontend: $$(cd $(FRONTEND_DIR) && cat package.json | grep version | head -1 | cut -d'"' -f4 2>/dev/null || echo 'Not setup')"
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo ""; \
		echo "🎮 GPU Information:"; \
		nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; \
	fi

ports:
	@echo "🔌 Checking used ports..."
	@for port in $(BACKEND_PORT) $(FRONTEND_PORT); do \
		if lsof -i:$$port >/dev/null 2>&1; then \
			echo "  Port $$port: IN USE"; \
			lsof -i:$$port | grep LISTEN; \
		else \
			echo "  Port $$port: AVAILABLE"; \
		fi; \
	done

deps:
	@echo "📦 Dependency Versions:"
	@echo "Backend Dependencies:"
	@. $(VENV_ACTIVATE) && \
	python -c "import sys; print(f'Python {sys.version}')" && \
	python -c "try: import fastapi; print(f'FastAPI {fastapi.__version__}'); except: pass" && \
	python -c "try: import torch; print(f'PyTorch {torch.__version__}'); except: pass" && \
	python -c "try: import transformers; print(f'Transformers {transformers.__version__}'); except: pass"
	@echo ""
	@echo "Frontend Dependencies:"
	@cd $(FRONTEND_DIR) && \
	echo "Node.js $$(node --version)" && \
	echo "npm $$(npm --version)" && \
	cat package.json | grep -E '"react"|"typescript"|"vite"' | head -3

# Utility targets
.PHONY: watch
watch:
	@echo "👀 Watching for changes..."
	@echo "Press Ctrl+C to stop"
	@while true; do \
		make status; \
		echo ""; \
		sleep 10; \
	done