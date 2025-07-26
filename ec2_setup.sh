#!/bin/bash
# ec2_setup.sh - Setup script for EC2 deployment

set -e

echo "🚀 Setting up IDP Pipeline on EC2..."

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
echo "🐳 Installing Docker..."
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Install Docker Compose
echo "🔧 Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install NVIDIA Docker (for GPU support)
echo "🎮 Installing NVIDIA Docker..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Add user to docker group
echo "👤 Adding user to Docker group..."
sudo usermod -aG docker $USER

# Create project directory
echo "📁 Creating project directory..."
mkdir -p ~/idp_pipeline
cd ~/idp_pipeline

echo "📋 Setup complete! Next steps:"
echo "1. Copy your project files to ~/idp_pipeline/"
echo "2. Run: cd ~/idp_pipeline && docker-compose up --build -d"
echo "3. Check logs: docker-compose logs -f"
echo "4. Access API at http://your-ec2-ip:8000"
echo "5. Monitor with Flower at http://your-ec2-ip:5555"

echo "🔐 Security Notes:"
echo "- Configure EC2 security groups to allow ports 8000, 5555"
echo "- Change SECRET_KEY in .env file"
echo "- Consider using SSL/TLS for production"
