#!/bin/bash

# SceneScriber AI Startup Script
# Starts both backend and frontend services with proper environment setup

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Trap for cleanup on script exit
cleanup() {
    log_info "Shutting down services..."
    
    # Kill background processes
    if [ -n "$BACKEND_PID" ]; then
        log_info "Stopping backend (PID: $BACKEND_PID)..."
        kill $BACKEND_PID 2>/dev/null || true
    fi
    
    if [ -n "$FRONTEND_PID" ]; then
        log_info "Stopping frontend (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    
    log_success "Cleanup complete!"
}

trap cleanup EXIT

# Check if setup has been run
check_setup() {
    log_info "Checking if setup has been completed..."
    
    # Check backend
    if [ ! -d "backend/venv" ]; then
        log_error "Backend virtual environment not found. Please run ./setup.sh first."
        exit 1
    fi
    
    if [ ! -f "backend/requirements-minimal.txt" ]; then
        log_error "Backend requirements file not found."
        exit 1
    fi
    
    # Check frontend
    if [ ! -d "frontend/node_modules" ]; then
        log_error "Frontend node_modules not found. Please run ./setup.sh first."
        exit 1
    fi
    
    log_success "Setup verification passed!"
}

# Start backend service
start_backend() {
    log_info "Starting backend service..."
    
    cd backend
    
    # Activate virtual environment
    if [ ! -f "venv/bin/activate" ]; then
        log_error "Virtual environment activation script not found."
        exit 1
    fi
    
    source venv/bin/activate
    
    # Check if required packages are installed
    if ! python -c "import fastapi" 2>/dev/null; then
        log_warning "FastAPI not found in virtual environment. Installing requirements..."
        pip install -r requirements-minimal.txt
    fi
    
    # Create necessary directories if they don't exist
    mkdir -p uploads exports
    
    # Start FastAPI server
    log_info "Launching FastAPI backend on http://localhost:8000"
    log_info "API documentation: http://localhost:8000/docs"
    
    # Run in background and capture PID
    uvicorn src.main:app --reload --port 8000 --host 0.0.0.0 &
    BACKEND_PID=$!
    
    # Wait a bit for server to start
    sleep 3
    
    # Check if backend is running
    if curl -s http://localhost:8000 > /dev/null; then
        log_success "Backend is running successfully!"
    else
        log_error "Backend failed to start. Check backend/server.log for details."
        exit 1
    fi
    
    cd ..
}

# Start frontend service
start_frontend() {
    log_info "Starting frontend service..."
    
    cd frontend
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        log_warning "node_modules not found. Installing dependencies..."
        npm install
    fi
    
    # Start React development server
    log_info "Launching React frontend on http://localhost:3000"
    
    # Run in background and capture PID
    npm run dev &
    FRONTEND_PID=$!
    
    # Wait a bit for server to start
    sleep 5
    
    # Check if frontend is running
    if curl -s http://localhost:3000 > /dev/null; then
        log_success "Frontend is running successfully!"
    else
        log_warning "Frontend might still be starting up. This can take a moment..."
        # Give it more time
        sleep 5
        if curl -s http://localhost:3000 > /dev/null; then
            log_success "Frontend is now running!"
        else
            log_error "Frontend failed to start. Check frontend/frontend.log for details."
            exit 1
        fi
    fi
    
    cd ..
}

# Display startup banner
show_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║                 🎬 SceneScriber AI                       ║"
    echo "║           Video Scene Analyzer Startup                   ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo "Starting both backend and frontend services..."
    echo ""
}

# Display access information
show_access_info() {
    echo ""
    echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}                  SERVICES STARTED!                       ${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${BLUE}🌐 Access Points:${NC}"
    echo -e "  • Frontend UI:    ${GREEN}http://localhost:3000${NC}"
    echo -e "  • Backend API:    ${GREEN}http://localhost:8000${NC}"
    echo -e "  • API Docs:       ${GREEN}http://localhost:8000/docs${NC}"
    echo ""
    echo -e "${BLUE}📊 Service Status:${NC}"
    echo -e "  • Backend PID:    ${GREEN}$BACKEND_PID${NC}"
    echo -e "  • Frontend PID:   ${GREEN}$FRONTEND_PID${NC}"
    echo ""
    echo -e "${BLUE}📝 Log Files:${NC}"
    echo -e "  • Backend logs:   ${YELLOW}backend/server.log${NC}"
    echo -e "  • Frontend logs:  ${YELLOW}frontend/frontend.log${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  Press Ctrl+C to stop all services${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Main execution
main() {
    show_banner
    
    # Check if setup is complete
    check_setup
    
    # Start services
    start_backend
    start_frontend
    
    # Show access information
    show_access_info
    
    # Keep script running and wait for Ctrl+C
    log_info "Services are running. Press Ctrl+C to stop."
    
    # Wait for both processes
    wait $BACKEND_PID $FRONTEND_PID
}

# Run main function
main "$@"