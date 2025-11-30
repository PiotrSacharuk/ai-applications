# MongoDB Atlas - Quick Setup (FREE)

## Step 1: Create MongoDB Atlas account
1. Go to: **https://www.mongodb.com/cloud/atlas/register**
2. Sign up (email + password or Google/GitHub)
3. Fill out brief survey or click "Skip"

## Step 2: Create FREE cluster (M0 Sandbox)
1. After logging in, click **"Create"** or **"Build a Database"**
2. Select **FREE** tier (M0 Sandbox):
   - Storage: 512 MB (sufficient for testing)
   - Shared RAM
   - **COMPLETELY FREE** forever
3. Select **Cloud Provider & Region**:
   - Provider: AWS (recommended if you already have AWS)
   - Region: **us-east-1** (N. Virginia) - closest to your S3
4. **Cluster Name**: `Cluster0` (leave default or change)
5. Click **"Create Deployment"** or **"Create Cluster"**
6. Wait ~3-5 minutes for cluster creation

## Step 3: Create database user
You'll see "Security Quickstart" popup:
1. **Username**: `admin` (or something else)
2. **Password**: Click **"Autogenerate Secure Password"** and **SAVE IT!**
   - Or enter your own password (save it!)
3. Click **"Create Database User"**

If popup didn't show:
1. In left menu: **Database Access**
2. Click **"Add New Database User"**
3. Authentication Method: **Password**
4. Username: `admin`
5. Password: Autogenerate (and save!) or enter your own
6. Database User Privileges: **Atlas admin** (or Read and write to any database)
7. Click **"Add User"**

## Step 4: Add network access
1. In left menu: **Network Access**
2. Click **"Add IP Address"**
3. Click **"Allow Access from Anywhere"** (for testing)
   - Adds: `0.0.0.0/0`
   - WARNING: In production, set specific IP!
4. Click **"Confirm"****

## Step 5: Get Connection String
1. In left menu: **Database** (database icon)
2. Next to your cluster, click **"Connect"**
3. Select: **"Drivers"** (or "Connect your application")
4. Driver: **Python**, Version: **3.12 or later**
5. Copy **Connection String**:
   ```
   mongodb+srv://admin:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```

## Step 6: Fill in Connection String
Replace `<password>` with REAL user password:
```
mongodb+srv://admin:YourPassword123@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority&ssl=true
```

## Step 7: Paste into .env
```bash
MONGODB_URL=mongodb+srv://admin:YourPassword123@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority&ssl=true
```

## Test connection (optional):
```bash
# Install pymongo if not already installed
poetry add pymongo

# Test in Python:
python3 -c "import pymongo; client = pymongo.MongoClient('YOUR_CONNECTION_STRING'); print('Connected!'); print(client.list_database_names())"
```

## Done!
Your free MongoDB Atlas is ready. Database and collection will be created automatically on first application run.

---

## FREE tier (M0) limits:
- 512 MB storage
- Shared RAM
- Unlimited connections (with per-second limits)
- **FREE FOREVER** (no trial period)
- Automatic daily backups
- Sufficient for small projects/testing
