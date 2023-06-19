import os, sys
from azure.cosmos.aio import CosmosClient as cosmos_client
from azure.cosmos import exceptions
from dotenv import dotenv_values

# load environment variables
config = ""
print("************************************************")
#print(os.getenv('ENVIRONMENT'))
print("************************************************")
# if os.getenv('BRANCH')=='DEV':
#     config = dotenv_values(
#         dotenv_path=os.path.join(os.getcwd(), "access_files", "dev.env")
#     )
# else:
#if os.environ['ENVIRONMENT']=='DEVELOPMENT':
config = dotenv_values(
dotenv_path=os.path.join(os.getcwd(), "access_files", "dev.env")
)
'''else :
config = dotenv_values(
dotenv_path=os.path.join(os.getcwd(), "access_files", "prod.env")
)'''


# <add_uri_and_key>
print(config.keys())
endpoint = config["COSMOS_DB_URI"]
key = config["COSMOS_DB_PRIMARY_KEY"]


# <define_database_and_container_name>
database_name = config["COSMOS_DB_NAME"]

# creating cosmos clinet for fetching table data
client = cosmos_client(endpoint, credential = key)  

# get database object here
async def get_db(client, database_name):
    try:
        database_obj  = client.get_database_client(database_name)
        await database_obj.read()
        print(f'Connected to {database_name}')
        print(database_obj)
        return database_obj
    except exceptions.CosmosResourceNotFoundError:
        print("Database not found..!!")

# Get the container in the database with the information 
async def get_container(container_name):
    try:        
        database_obj = await get_db(client, database_name)
        container_items = database_obj.get_container_client(container_name)
        await container_items.read() 
        return container_items
    except exceptions.CosmosHttpResponseError:
        raise exceptions.CosmosHttpResponseError

async def run(container_name):
    container_items = await get_container(container_name)
    #await read_items(container_items,"IT")
    # query="select * from Temp_ConfigTable"
    # await query_items(container_items,query)
    # IT_ticker_id , IT_ticker_name = get_ticker_names_and_id('IT', items)
    # print(IT_ticker_id)

    
    return container_items
