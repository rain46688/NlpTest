import boto3
import pandas as pd
import io
import warnings
import configparser

# string = '/pub/login' # 1863개
# string = '/pub/loginToken' # 0개
string = '/svc/refreshToken' # 37개

# 경고 메시지 무시
warnings.filterwarnings(action='ignore', category=UserWarning, module='openpyxl')

config = configparser.ConfigParser()
config.read('./config.ini')

aws_access_key_id = config['AWS']['aws_access_key_id'].strip()
aws_secret_access_key = config['AWS']['aws_secret_access_key'].strip()
region_name = config['AWS']['region_name'].strip()
bucket_name = config['AWS']['bucket_name'].strip()

# S3 클라이언트 설정
s3 = boto3.client('s3', aws_access_key_id=id, aws_secret_access_key=key, region_name=region_name)

# 버킷에서 모든 객체를 가져옵니다.
response = s3.list_objects(Bucket=bucket_name, Prefix='useLog/release/')

# 'useLog_날짜.xlsx' 형식의 모든 파일을 가져옵니다.
files = [item['Key'] for item in response['Contents'] if 'useLog_' in item['Key'] and item['Key'].endswith('.xlsx')]
total_count = 0

for file in files:
    print("파일명 : ", file)
    file_data = s3.get_object(Bucket=bucket_name, Key=file)
    data = pd.read_excel(io.BytesIO(file_data['Body'].read()))
    count = data['control'].str.contains(string).sum()
    total_count += count

print(f"' 총 {total_count}개")