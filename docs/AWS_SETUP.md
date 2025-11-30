# AWS Configuration for docu_chat.py

## Option 1: Using AWS (recommended for production)

### Step 1: Install AWS CLI
```bash
# Linux/WSL:
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# macOS:
brew install awscli

# Windows:
# Download installer from: https://awscli.amazonaws.com/AWSCLIV2.msi
```

### Step 2: Create AWS account (if you don't have one)
1. Go to https://aws.amazon.com/
2. Click "Create an AWS Account"
3. Fill out form (credit card required, but Free Tier is free)

### Step 3: Create IAM User with S3 access
1. Log in to AWS Console: https://console.aws.amazon.com/
2. Go to **IAM** (Identity and Access Management)
3. Click **Users** → **Add users**
4. Username: `docu-chat-app`
5. Check: **Access key - Programmatic access**
6. Permissions: **Attach existing policies directly**
   - Find and select: `AmazonS3FullAccess`
7. Click **Next** → **Create user**
8. **SAVE CREDENTIALS** (Access Key ID and Secret Access Key) - you won't see them again!

### Step 4: Create S3 Bucket
```bash
# After configuring AWS CLI:
aws configure
# Paste Access Key ID
# Paste Secret Access Key
# Region: us-east-1
# Output format: json

# Create bucket (name must be globally unique):
aws s3 mb s3://docu-chat-YOUR-NAME-12345
aws s3api put-object --bucket docu-chat-YOUR-NAME-12345 --key documents/
```

Or in AWS Console:
1. Go to **S3**: https://s3.console.aws.amazon.com/
2. Click **Create bucket**
3. Name: `docu-chat-your-name` (must be globally unique)
4. Region: `us-east-1`
5. **Unblock** "Block all public access" if you want (optional)
6. Click **Create bucket**
7. Open bucket → **Create folder** → name: `documents`

### Step 5: Fill in .env
```bash
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE  # Your Access Key ID
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY  # Your Secret
AWS_S3_BUCKET=docu-chat-your-name  # Bucket name
AWS_S3_REGION=us-east-1
AWS_S3_PATH=documents/
```

---

## Option 2: Local file storage (for testing, WITHOUT AWS)

If you want to test without AWS, I can modify the code to use local file system instead of S3.

### Advantages of local solution:
- No AWS costs
- Immediate start without cloud configuration
- Easier debugging

### Disadvantages:
- Doesn't scale (single server only)
- No cloud backup
- Not suitable for production

**Do you want to:**
1. **Configure AWS** (recommended, Free Tier gives 5GB S3 for free)
2. **Modify code for local files** (quick start for testing)

---

## AWS Free Tier cost estimate:
- **S3**: 5 GB storage, 20,000 GET requests, 2,000 PUT requests monthly - **FREE**
- **After Free Tier**: ~$0.023/GB monthly
- For small test application: **practically $0**

---

## Next Steps

After configuring AWS, see [MONGODB_SETUP.md](MONGODB_SETUP.md) for MongoDB Atlas configuration.
