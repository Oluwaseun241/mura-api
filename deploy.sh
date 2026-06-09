#!/bin/bash

# West African Food Classifier Deployment Script
# This script helps deploy the food classifier API

set -e

echo "🍽️ West African Food Classifier Deployment Script"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required files exist
check_requirements() {
    print_status "Checking requirements..."
    
    if [ ! -f "food_classifier.pth" ]; then
        print_error "Model file 'food_classifier.pth' not found!"
        print_error "Please run 'python train.py' first to train the model."
        exit 1
    fi
    
    if [ ! -d "images" ]; then
        print_error "Images directory not found!"
        exit 1
    fi
    
    print_status "✅ All requirements met"
}

# Install dependencies
install_deps() {
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    print_status "✅ Dependencies installed"
}

# Build Docker image
build_docker() {
    print_status "Building Docker image..."
    docker build -t food-classifier:latest .
    print_status "✅ Docker image built successfully"
}

# Run with Docker Compose
run_docker_compose() {
    print_status "Starting services with Docker Compose..."
    docker-compose up -d
    print_status "✅ Services started"
    print_status "API available at: http://localhost:8000"
    print_status "Documentation at: http://localhost:8000/docs"
}

# Run locally
run_local() {
    print_status "Starting API locally..."
    python app.py
}

# Test the API
test_api() {
    print_status "Testing API..."
    
    # Wait for API to start
    sleep 5
    
    # Test health endpoint
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_status "✅ API is healthy"
    else
        print_warning "⚠️ API health check failed"
    fi
    
    # Test with sample image if available
    if [ -f "test/j1.jpeg" ]; then
        print_status "Testing with sample image..."
        curl -X POST "http://localhost:8000/predict" \
             -F "file=@test/j1.jpeg" \
             -F "top_k=3" \
             -H "Accept: application/json" | jq . || print_warning "jq not available, raw response shown"
    fi
}

# Show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  local     Run locally with Python"
    echo "  docker    Build and run with Docker"
    echo "  compose   Run with Docker Compose"
    echo "  test      Test the running API"
    echo "  stop      Stop Docker services"
    echo "  logs      Show Docker logs"
    echo "  clean     Clean up Docker resources"
    echo ""
    echo "Examples:"
    echo "  $0 local     # Run locally"
    echo "  $0 docker    # Run with Docker"
    echo "  $0 compose   # Run with Docker Compose"
}

# Stop services
stop_services() {
    print_status "Stopping services..."
    docker-compose down
    print_status "✅ Services stopped"
}

# Show logs
show_logs() {
    print_status "Showing logs..."
    docker-compose logs -f
}

# Clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker-compose down -v
    docker system prune -f
    print_status "✅ Cleanup complete"
}

# Main script logic
case "${1:-help}" in
    "local")
        check_requirements
        install_deps
        run_local
        ;;
    "docker")
        check_requirements
        build_docker
        docker run -p 8000:8000 --env-file .env food-classifier:latest
        ;;
    "compose")
        check_requirements
        run_docker_compose
        test_api
        ;;
    "test")
        test_api
        ;;
    "stop")
        stop_services
        ;;
    "logs")
        show_logs
        ;;
    "clean")
        cleanup
        ;;
    "help"|*)
        show_usage
        ;;
esac
