import os, sys
from dotenv import dotenv_values
from azure.identity import DefaultAzureCredential 
from azure.keyvault.secrets import SecretClient

# load environment variables
# print("SYS PATH", sys.path[0], " NEW LINE \n ", os.path.pardir)


# if os.environ['ENVIRONMENT']=='DEVELOPMENT':
config = dotenv_values(
dotenv_path=os.path.join(os.getcwd(), "access_files", "dev.env")
)
'''else :
    config = dotenv_values(
    dotenv_path=os.path.join(os.getcwd(), "access_files", "prod.env")
)'''

# get the client authentication which is mentioned in the CLI ENV variables
credential = DefaultAzureCredential()


def retrieveSecrets():
    """_summary_

    Returns:
        Dictiionary: It has all the key value pairs of key vault of azure
    """
    secretDictValue = {}
    print("PRINTING CONFIG")
    for key, value in config.items():
        print(key, " " , value)
    print("PRINTED CONFIG")
    vaultName = config["VAULT_NAME"]
    secretClient = SecretClient(vault_url=f"https://{vaultName}.vault.azure.net/", credential=credential)
    secretProperties = secretClient.list_properties_of_secrets()
    for secretProperty in secretProperties:
        print(secretProperty.name)
        secretDictValue[secretProperty.name] = secretClient.get_secret(secretProperty.name).value

    return secretDictValue