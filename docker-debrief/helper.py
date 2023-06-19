import simplejson as json
import boto3
import time
# import config
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO)


logging.warning('helper.py')

from access_file import get_secret
# region_name = "ap-south-1"
# def get_secret():

#     secret_name = "FABRIC-RDS-Secret-Keys"
#     

#     # Create a Secrets Manager client
#     session = boto3.session.Session()
#     client = session.client(
#         service_name="secretsmanager", region_name=region_name)

#     get_secret_value_response = client.get_secret_value(SecretId=secret_name)
#     logging.warning('helper.py get_secret')
#     return get_secret_value_response


# Creating the global variables
get=get_secret()
credentials = json.loads(get["SecretString"])
print(credentials)
default_bucket = credentials["default_bucket"]
region_name=credentials["region_name"]
# default_bucket = "algoanalytics-fabric-website"
DISTRIBUTION_ID = credentials["cloudfront_distribution_id"]


cf = boto3.client('cloudfront',region_name=region_name)
# aws_access_key_id=config.algo_access_key,aws_secret_access_key=config.algo_secret_access_token,region_name=config.region_name)
logging.warning('helper.py Creating the global variables')


def create_invalidation():
    res = cf.create_invalidation(
        DistributionId=DISTRIBUTION_ID,
        InvalidationBatch={
            'Paths': {
                'Quantity': 1,
                'Items': [
                    '/Json/*'
                ]
            },
            'CallerReference': str(time.time()).replace(".", "")
        }
    )
    invalidation_id = res['Invalidation']['Id']
    logging.warning('helper.py create_invalidation')
    return invalidation_id


def myconverter(o):
    if isinstance(o, datetime):
        logging.warning('helper.py myconverter')
        return o.__str__()


def write_json_file(file_name, data):
    """Write json object to json file
    Args:
        file_name (sting): filename (currenty ticker symbols)
        data (dict): dict as json object to write to file
    """
    
    session = boto3.session.Session()
    client = session.client(service_name="s3", region_name=region_name)
    #  aws_access_key_id=config.algo_access_key, aws_secret_access_key=config.algo_secret_access_token)
    client.put_object(
        Body=json.dumps(data,default=str), Bucket=default_bucket, Key='Json/{}'.format(file_name))
    
    if "summary" in file_name:
        id = create_invalidation()
        print("Invalidation created successfully with Id: " + id)
    logging.warning('helper.py write_json_file')
