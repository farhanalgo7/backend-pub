import boto3
# import config
region_name = "ap-south-1"

def get_secret():

    secret_name = "FABRIC-RDS-Secret-Keys"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)
    #  aws_access_key_id=config.algo_access_key ,
    # endpoint_url="http://dynamodb.ap-south-1.amazonaws.com",
    # aws_secret_access_key=config.algo_secret_access_token,
    # region_name=config.region_name)

    get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    return get_secret_value_response



