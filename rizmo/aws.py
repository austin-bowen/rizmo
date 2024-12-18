import boto3
from mypy_boto3_polly import PollyClient

from rizmo import secrets
from rizmo.config import config


def get_polly_client() -> PollyClient:
    session = get_aws_session()
    return session.client('polly')


def get_aws_session() -> boto3.Session:
    return boto3.Session(
        aws_access_key_id=secrets.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=secrets.AWS_SECRET_ACCESS_KEY,
        region_name=config.aws_region,
    )
