import pandas as pd
import boto3
from io import BytesIO
import warnings
import configparser

# 경고 메시지 무시
warnings.filterwarnings(action='ignore', category=UserWarning, module='openpyxl')

config = configparser.ConfigParser()
config.read('./config.ini')

id = config['AWS']['aws_access_key_id'].strip()
key = config['AWS']['aws_secret_access_key'].strip()
region_name = config['AWS']['region_name'].strip()
bucket_name = config['AWS']['bucket_name'].strip()

# S3 클라이언트 설정
s3 = boto3.client('s3', aws_access_key_id=id, aws_secret_access_key=key, region_name=region_name)

# 버킷에서 모든 객체를 가져옵니다.
response = s3.list_objects(Bucket=bucket_name, Prefix='useLog/release/')

# 'useLog_날짜.xlsx' 형식의 모든 파일을 가져옵니다.
files = [item['Key'] for item in response['Contents'] if 'useLog_' in item['Key'] and item['Key'].endswith('.xlsx')]

# 결과를 저장할 DataFrame을 생성합니다.
result = pd.DataFrame()

# 각 파일을 처리합니다.
for file in files:
    print("파일명 : ", file)
    # 파일을 로드합니다.
    obj = s3.get_object(Bucket=bucket_name, Key=file)
    df = pd.read_excel(BytesIO(obj['Body'].read()))

    # 'control' 컬럼이 '/pub/likeSearch'인 행을 찾습니다.
    df = df[df['control'] == '/pub/likeSearch']

    # 'logstring' 컬럼에서 사용자 이름과 검색어를 추출합니다.
    df['name'] = df['logstring'].str.extract('(.+) :')
    df['search'] = df['logstring'].str.extract('search=([가-힣 ]+),')
    
    # NaN 값을 제거합니다.
    df = df.dropna()

    # 사용자별 검색어 데이터를 생성합니다.
    grouped = df.groupby('name')['search'].apply(lambda x: ', '.join(set(x))).reset_index()

    # 결과를 result DataFrame에 추가합니다.
    result = pd.concat([result, grouped])

# 결과를 새 엑셀 파일로 저장합니다.
result.to_excel('result.xlsx', index=False)