import os
import pymysql
from dotenv import dotenv_values

# custom imports
from .vault import retrieveSecrets

# load environment variables
config = ""
print("****************************************")

print(os.environ['ENVIRONMENT'])


print("****************************************")
if os.environ['ENVIRONMENT']=='DEVELOPMENT':
    config = dotenv_values("dev.env") 
else :
    config = dotenv_values("prod.env")

# retrieve secrets from the azure key vault
credential = retrieveSecrets()


def get_connection():
    # Obtain connection string information from the portal
    sql_config = {
        'host':
        credential["host"],
        'user':
        f'{credential["azure-sql-username"]}'
,
        
        'password':
        credential["azure-sql-password"],
        'database':
        credential["Database-Name"],
        'ssl_ca':
        os.path.join(os.getcwd(
        ), "access_files", "certs", "DigiCertGlobalRootG2.crt.pem").replace(
            "\\", "\\\\"
        )  #"E:\\InternWork\\Algoanalytics\\fabric-backend\\docker-stock-ingestion\\access_files\\certs\\DigiCertGlobalRootG2.crt.pem"
    }

    connection = pymysql.connect(**sql_config)
    print("Connection established..!!")
    return connection
