# GitHub Actions Deployment Guide

## Overview
This workflow automatically deploys your GA Framework application to all AWS EC2 instances managed by Terraform whenever you push to `main` or `development` branches.

## How It Works

### 1. **Terraform Integration**
- Connects to AWS and reads your existing Terraform state
- Automatically retrieves EC2 instance IPs from Terraform outputs
- No need to manually configure IP addresses!

### 2. **Multi-Instance Deployment**
Deploys to all 3 instances:
- ✅ Public Instance (main application server)
- ✅ Private Instance 1 (worker node)
- ✅ Private Instance 2 (worker node)

### 3. **Deployment Steps**
For each instance:
1. Creates application directory (`/home/ubuntu/ga-framework`)
2. Copies all application files via rsync
3. Installs Python dependencies in virtual environment
4. Sets up Docker (optional)
5. Starts the application
6. Runs health checks

## Setup Instructions

### Step 1: Configure GitHub Secrets

Go to your repository: **Settings → Secrets and variables → Actions → New repository secret**

Add these 3 secrets:

#### 1. `AWS_ACCESS_KEY_ID`
Your AWS access key with permissions to:
- Read EC2 instances
- Access Terraform state (if stored in S3)

```
Example: AKIAIOSFODNN7EXAMPLE
```

#### 2. `AWS_SECRET_ACCESS_KEY`
Your AWS secret access key (corresponding to the access key above)

```
Example: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

#### 3. `EC2_SSH_KEY`
The **FULL CONTENT** of your `my-default-testing-key.pem` file

```
-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEAw7QamKmP8dCfJJW4wyY...
... (all your key content) ...
... (don't skip any lines) ...
-----END RSA PRIVATE KEY-----
```

⚠️ **Important:** Copy the ENTIRE file content including the BEGIN and END lines!

### Step 2: Verify Terraform Configuration

Make sure your Terraform outputs are configured (already done in your `infrastructure/outputs.tf`):

```terraform
output "public_instance_ip" { ... }
output "private_instance_public_ip" { ... }
output "private_instance_2_public_ip" { ... }
```

### Step 3: Update Security Groups (if needed)

Your EC2 security groups must allow SSH from GitHub Actions runners.

**Option 1: Allow from anywhere (simpler, less secure)**
```terraform
ingress {
  from_port   = 22
  to_port     = 22
  protocol    = "tcp"
  cidr_blocks = ["0.0.0.0/0"]  # Allow SSH from anywhere
}
```

**Option 2: Allow from GitHub Actions IP ranges (more secure)**
- Get IP ranges from: https://api.github.com/meta
- Add them to your security group

### Step 4: Test the Workflow

1. **Commit and push** any change to `main` or `development` branch:
   ```bash
   git add .
   git commit -m "Test deployment workflow"
   git push origin main
   ```

2. **Monitor the workflow:**
   - Go to **Actions** tab in your GitHub repository
   - Click on the running workflow
   - Watch the deployment progress

3. **Check logs** for each step to ensure success

## Workflow Triggers

The workflow runs when:
- ✅ Push to `main` branch
- ✅ Push to `development` branch
- ✅ Pull request to `main` branch

## What Gets Deployed

**Included:**
- ✅ All Python source files (`src/`)
- ✅ Configuration files (`requirements.txt`, etc.)
- ✅ Scripts (`scripts/`)
- ✅ Documentation (`docs/`)
- ✅ Docker configuration (`infrastructure/docker/`)

**Excluded (for safety):**
- ❌ `.git` directory
- ❌ `__pycache__` files
- ❌ `.pyc` files
- ❌ `.github` workflows
- ❌ Terraform state files
- ❌ `.pem` key files

## Troubleshooting

### Issue: "Permission denied (publickey)"
**Solution:** 
- Verify `EC2_SSH_KEY` secret contains the complete private key
- Check that the key corresponds to the one configured in Terraform (`var.key_name`)

### Issue: "Terraform init failed"
**Solution:**
- Ensure AWS credentials have permission to read Terraform state
- If using S3 backend, verify `terraform.tfvars` configuration

### Issue: "Host key verification failed"
**Solution:**
- The workflow handles this automatically with `ssh-keyscan`
- If persists, check if EC2 instances are running

### Issue: "Cannot connect to EC2 instance"
**Solution:**
- Verify security group allows SSH (port 22)
- Check if instances are running: `terraform show`
- Verify public IPs are assigned: `terraform output`

### Issue: "Python dependencies installation failed"
**Solution:**
- SSH into instance manually and check disk space: `df -h`
- Verify `requirements.txt` has correct package names
- Check Python version compatibility

## Advanced Configuration

### Customize Application Directory
Edit the workflow file and change:
```yaml
env:
  APP_DIR: /home/ubuntu/your-custom-path
```

### Change AWS Region
Already set to `ap-south-1` (matching your Terraform), but you can change:
```yaml
env:
  AWS_REGION: us-east-1
```

### Deploy to Specific Instances Only
Comment out unwanted deployment jobs in the workflow:
```yaml
# - name: Deploy to Private Instance 2  # Disabled
#   run: |
#     ...
```

### Add Post-Deployment Tests
Add a new step after deployment:
```yaml
- name: Run integration tests
  run: |
    ssh -i ~/.ssh/deploy_key.pem ubuntu@${{ steps.terraform.outputs.public_instance_ip }} << 'EOF'
      cd /home/ubuntu/ga-framework
      source venv/bin/activate
      python -m pytest tests/
    EOF
```

## Monitoring Deployments

### View Deployment Status
- GitHub Actions UI shows real-time progress
- Each step is clearly labeled with emojis (🚀, ✅, 🐳, etc.)
- Failed steps are highlighted in red

### SSH into Instances After Deployment
```bash
# Public instance
ssh -i infrastructure/my-default-testing-key.pem ubuntu@<PUBLIC_IP>

# Check application
cd /home/ubuntu/ga-framework
source venv/bin/activate
python --version
pip list
```

## Security Best Practices

✅ **Do:**
- Store sensitive data in GitHub Secrets
- Use restricted AWS IAM roles
- Regularly rotate AWS access keys
- Keep SSH keys secure
- Review workflow logs for sensitive data exposure

❌ **Don't:**
- Commit secrets to the repository
- Share AWS credentials
- Allow unrestricted SSH access (0.0.0.0/0) in production
- Leave debug output that exposes sensitive data

## CI/CD Pipeline Flow

```
┌─────────────────┐
│  Git Push       │
│  (main/dev)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ GitHub Actions  │
│ Workflow Start  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Setup & Config  │
│ - Python        │
│ - AWS creds     │
│ - Terraform     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Get EC2 IPs     │
│ from Terraform  │
└────────┬────────┘
         │
         ├──────────┬──────────┐
         ▼          ▼          ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │Public  │ │Private1│ │Private2│
    │Instance│ │Instance│ │Instance│
    └───┬────┘ └───┬────┘ └───┬────┘
        │          │          │
        ▼          ▼          ▼
    Deploy    Deploy    Deploy
    Install   Install   Install
    Start     Start     Start
    Health    Health    Health
        │          │          │
        └──────────┴──────────┘
                   │
                   ▼
            ┌──────────────┐
            │  Summary &   │
            │  Cleanup     │
            └──────────────┘
```

## Support

For issues or questions:
1. Check workflow logs in GitHub Actions
2. Review this README
3. Check Terraform state: `terraform show`
4. Verify AWS connectivity: `aws ec2 describe-instances`

---

**Last Updated:** October 21, 2025
**Repository:** GA-Framework-for-Parameter-Tuning
