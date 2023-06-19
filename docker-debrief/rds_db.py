import mysql.connector
import os
from access_file import get_secret
import json
import boto3
# import config
import pymysql

# retrieve credentials
get=get_secret()
credentials = json.loads(get["SecretString"])
print(credentials)

database = credentials["Database_Name"]
user = credentials["username"]
region = "ap-south-1"

""" Connection to rds instance  """
client = boto3.client("rds", region_name=region)
# ,aws_access_key_id=config.algo_access_key,
# endpoint_url=&quot;http://dynamodb.ap-south-1.amazonaws.com&quot;,
# aws_secret_access_key=config.algo_secret_access_token)
os.environ["LIBMYSQL_ENABLE_CLEARTEXT_PLUGIN"] = "1"
host = "fabric-database-1.cptxszv3ahgy.ap-south-1.rds.amazonaws.com"
port = 3306

def rds_con():
    token = client.generate_db_auth_token(DBHostname=host, Port="3306", DBUsername="fabric_user", Region=region)
    print("TOKEN IS:" ,token)
    # con=mysql.connector.connect(host=host, user="fabric_user", password=token)
    con = mysql.connector.connect(host=host, user=user, password=token, port=port, database=database)
    cursor=con.cursor()
    return cursor
# def get_connection():
#     # Obtain connection string information from the portal
#     sql_config = {
#         'host':
#         credential["host"],
#         'user':
#         f'{credential["username"]}'
# ,
        
#         'password':
#         credential["azure-sql-password"],
#         'database':
#         credential["Database-Name"],
#         'ssl_ca':
#         os.path.join(os.getcwd(
#         ), "access_files", "certs", "DigiCertGlobalRootG2.crt.pem").replace(
#             "\\", "\\\\"
#         )  #"E:\\InternWork\\Algoanalytics\\fabric-backend\\docker-stock-ingestion\\access_files\\certs\\DigiCertGlobalRootG2.crt.pem"
#     }

#     connection = pymysql.connect(**sql_config)
#     print("Connection established..!!")
#     return connection
