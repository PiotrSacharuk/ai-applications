"""
AWS S3 Storage Implementation
"""
import os
import boto3
import awswrangler as wr
from ...interfaces import StorageProvider
from ...config import S3_KEY, S3_SECRET, S3_REGION, S3_BUCKET, S3_PATH


class S3Storage(StorageProvider):
    """AWS S3 storage implementation"""

    def __init__(self):
        self.session = boto3.Session(
            aws_access_key_id=S3_KEY,
            aws_secret_access_key=S3_SECRET,
            region_name=S3_REGION,
        )
        self.bucket = S3_BUCKET
        self.path = S3_PATH

    def download_document(self, filename: str) -> str:
        """Download document from S3 to local file"""
        local_file = filename.split("/")[-1]
        wr.s3.download(
            path=f"s3://{self.bucket}/{self.path}{filename}",
            local_file=local_file,
            boto3_session=self.session
        )
        return local_file

    def upload_document(self, local_path: str) -> str:
        """Upload document to S3 and return S3 path"""
        filename = local_path.split("/")[-1]
        s3_path = f"s3://{self.bucket}/{self.path}{filename}"
        wr.s3.upload(
            local_file=local_path,
            path=s3_path,
            boto3_session=self.session,
        )
        return s3_path
