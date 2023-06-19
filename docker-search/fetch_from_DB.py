import os
import json
import argparse
import time
import datetime as dt
import uuid
import logging

from boto3.dynamodb.conditions import Key, Attr

from utils import (
    check_type,
    get_dynamodb_connection,
    get_dynamodb_table,
    get_today,
    months_between
)

from update_embeddings import update_embeddings

from dotenv import load_dotenv
load_dotenv()

# Set logging level of Haystack to warning
logging.getLogger("haystack").setLevel(logging.WARNING)

# Set logging level of Azure to Warning to avoid unnecessary HTTP logs
Azure_logger = logging.getLogger('azure.core.pipeline.policies.http_logging_policy')
Azure_logger.setLevel(logging.WARNING)

def fetch_data(table, type="news"):
    """Fetch data from the DynamoDB, between the from and to dates sent in
    as command line arguments, and return the list of items.

    Args:
        table (dynamodb.Table): DynamoDB Table, either "News_Table" or "Tweets_Table"
        type (str, optional): Type, either "news" or "tweets". Defaults to "news".

    Returns:
        list: List of items (row entries) from the database
    """
    global from_date, to_date, from_month, to_month, from_datetime, to_datetime
    
    check_type(type)
    
    month_key = f"{type}_month"
    ner_flag = f"{type}_ner_flag"
    date_key = f"{type}_date"
    
    if type == "tweets":
        # Correct to tweet instead of tweets
        ner_flag = "tweet_ner_flag"
        date_key = "tweet_date"
    
    items = []
    
    # There's a limit to how much data can be fetched at once
    # So need to paginate queries
    # https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Query.Pagination.html 
    limit = 50
    
    # We filter by date being between our requirement, and ner_flag = True
    filter_expression = Attr(date_key).between(from_datetime, to_datetime) & Attr(ner_flag).eq(True)
    
    # Sleep times
    normal_wait_time = 0.1  # in seconds
    
    # Sleep more if normal fetch fails
    failure_wait_time = 5   # in seconds
    
    for month_date in months_between(from_date, to_date):
        # month_date is a datetime.date object. Format to YYYY-MM
        month = month_date.strftime("%Y-%m")
        
        # Key Condition can only be an equality, so need to loop over months
        key_condition_expression = Key(month_key).eq(month)
        
        # First query
        response = table.query(
            Limit=limit,
            KeyConditionExpression=key_condition_expression,
            FilterExpression=filter_expression
        )
        
        items.extend(response['Items'])
        
        # Keep querying for current month till done
        while 'LastEvaluatedKey' in response:
            try:
                print(f"Fetching {type} items. Current Length = {len(items)}")
                time.sleep(normal_wait_time)
                response = table.query(
                    Limit=limit,
                    ExclusiveStartKey=response['LastEvaluatedKey'],
                    KeyConditionExpression=key_condition_expression,
                    FilterExpression=filter_expression
                )
                items.extend(response['Items'])
            except:
                time.sleep(failure_wait_time)
                response = table.query(
                    Limit=limit,
                    ExclusiveStartKey=response['LastEvaluatedKey'],
                    KeyConditionExpression=key_condition_expression,
                    FilterExpression=filter_expression
                )
                items.extend(response['Items'])
    
    print(f"All {type} items fetched. Total Items: {len(items)}")
    return items


def remove_duplicates(item_list, type="news"):
    """Compare fetched data with the existing mappings, and remove duplicates.

    Args:
        item_list (list): List of items fetched from the database.
        type (str, optional): Type, either "news" or "tweets". Defaults to "news".

    Returns:
        list: List of items left after removing duplicates (i.e. items already present)
    """
    check_type(type)
    
    # Set Mapping file path
    mapping_file = os.path.join(os.environ['MAPPING_DIR'], f"{type}_mapping.json")
    item_id = f"{type}_ID"
    
    if type == "news":
        item_text = "long_description"
    else:
        item_text = "tweet_text"
    
    with open(mapping_file, "r", encoding="utf-8") as fp:
        mapping = json.load(fp)
    
    # Mapping file looks like UUID: [item_ID, item_month]
    # Get list of item_IDs reverse mapped from UUIDs
    existing_ids = [mapping[uid][0] for uid in mapping]

    # Remove duplicates, and items which don't have
    # long_description for news, and tweet_text for tweets
    new_items = [item for item in item_list if item[item_id] not in existing_ids and item[item_text]]
    
    print("Removed Duplicates.")
    print(f"Number of new {type.title()} = {len(new_items)}")
    return new_items


def assign_unique_ids(item_list, type="news"):
    """Assign UUIDs to the items, and update the mapping file.

    Args:
        item_list (list): List of items fetched from the database, after removing duplicates.
        type (str, optional): Type, either "news" or "tweets". Defaults to "news".
    """
    check_type(type)
    
    item_id = f"{type}_ID"
    item_date = f"{type}_date"
    mapping_file = os.path.join(os.environ['MAPPING_DIR'], f"{type}_mapping.json")
    
    if type == "tweets":
        item_date = "tweet_date"
    
    # Load existing mapping
    with open(mapping_file, "r", encoding="utf-8") as fp:
        mapping = json.load(fp)
    
    new_mapping = dict()
    for item in item_list:
        unique_id = str(uuid.uuid4())
        item['UUID'] = unique_id
        new_mapping[unique_id] = [item[item_id], item[item_date]]
    
    print(f"Updating Mapping for {type}")
    print(f"Length before update: {len(mapping)}")
    mapping.update(new_mapping)
    print(f"Length after update: {len(mapping)}")
    
    with open(mapping_file, "w", encoding="utf-8") as fp:
        json.dump(mapping, fp, indent=2)


def save_items_to_text_files(item_list, type="news"):
    """Save items as text documents in NEWS_DIR or TWEETS_DIR, depending on type.

    Args:
        item_list (list): List of items fetched from database, after removing duplicates and assigning UUIDS.
        type (str, optional): Type, either "news" or "tweets". Defaults to "news".
    """
    check_type(type)
    
    # File directory to save is NEWS_DIR or TWEETS_DIR, depending on type
    file_dir = os.environ[f"{type.upper()}_DIR"]
    
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
        
    if type == "news":
        item_text = "long_description"
    else:
        item_text = "tweet_text"
    
    print(f"Writing {type} to text documents.")
    for item in item_list:
        # Sometimes item['long_description'] or item['tweet_text'] can be False
        with open(os.path.join(file_dir, f"doc_{item['UUID']}.txt"), 'w', encoding="utf-8") as f:
            content = item[item_text].replace("\n", " ")
            f.write(content)
            

# -------------------------------------------------

if __name__ == "__main__":
    # Parse two arguments
    # --from-date YYYY-MM-DD
    # --to-date YYYY-MM-DD
    # We fetch data between these two dates

    parser = argparse.ArgumentParser(
        description="Fetch data from DynamoDB and update embeddings for DocumentStore",
        allow_abbrev=False
    )


    # --from-date or --from, parse as datetime.date
    parser.add_argument(
        "--from-date", "--from", 
        help="From date, in YYYY-MM-DD format (ISO 8601)",
        type=dt.date.fromisoformat,
    )

    # --to-date or --to, parse as datetime.date
    parser.add_argument(
        "--to-date", "--to",
        help="To date, in YYYY-MM-DD format (ISO 8601)",
        type=dt.date.fromisoformat,
    )
    
    # Get today as datetime.date
    today = get_today()
    remove_threshold = today - dt.timedelta(days=30)
    
    args = parser.parse_args()
    
    # Set from_date as yesterday if argument not given
    from_date = args.from_date or (dt.date.today() - dt.timedelta(days=1))
    to_date = args.to_date or today
    
    # Get months in YYYY-MM format
    from_month = from_date.strftime("%Y-%m")
    to_month = to_date.strftime("%Y-%m")

    # Get datetime in YYYY-MM-DD HH:MM:SS format
    from_datetime = from_date.isoformat() + " 00:00:00"
    to_datetime = to_date.isoformat() + " 23:59:59"

    dynamodb = get_dynamodb_connection()

    for type in ["news", "tweets"]:
        table = get_dynamodb_table(dynamodb, type)
        items = fetch_data(table, type=type)
        items = remove_duplicates(items, type=type)
        
        if len(items):
            assign_unique_ids(items, type=type)
            save_items_to_text_files(items, type=type)
            update_embeddings(type=type)
