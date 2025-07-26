# 🚀 EC2 Deployment Checklist

## Pre-Deployment (Local)
- [ ] Review and update .env file with your specific settings
- [ ] Change SECRET_KEY in .env to a secure value
- [ ] Verify all file paths and configurations

## EC2 Instance Setup (g4dn.2xlarge)
- [ ] Launch EC2 instance with Deep Learning AMI (Ubuntu 22.04)
- [ ] Configure security groups (ports 8000, 5555, 22)
- [ ] SSH into the instance
- [ ] Run: `./ec2_setup.sh` to install Docker and NVIDIA Docker

## Deploy Application
- [ ] Upload all project files to ~/idp_pipeline/
- [ ] Create directories: `mkdir -p models uploads logs`
- [ ] Update .env file if needed
- [ ] Build and start: `docker-compose up --build -d`
- [ ] Check status: `docker-compose ps`
- [ ] View logs: `docker-compose logs -f`

## Test Deployment
- [ ] Health check: `curl http://localhost:8000/health`
- [ ] Upload test PDF via API
- [ ] Monitor via Flower: `http://your-ec2-ip:5555`
- [ ] Check processing results

## Production Readiness
- [ ] Configure proper SSL/TLS certificates
- [ ] Set up monitoring and alerting
- [ ] Configure backup for models and data
- [ ] Set up log rotation
- [ ] Configure firewall rules
- [ ] Set up auto-scaling if needed

## Performance Tuning
- [ ] Adjust worker concurrency based on GPU memory
- [ ] Monitor GPU utilization with `nvidia-smi`
- [ ] Scale workers: `docker-compose up --scale text_worker=2 -d`
- [ ] Tune confidence thresholds in .env

Remember: The system is designed to handle 12-page proposal forms efficiently!
