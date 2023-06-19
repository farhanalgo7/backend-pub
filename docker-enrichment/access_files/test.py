from dotenv import dotenv_values
import sys, os

# load environment variables
# print("SYS PATH", sys.path[0], " NEW LINE \n ", os.path.pardir)
config = dotenv_values(
	dotenv_path=os.path.join(os.getcwd(), "access_files", ".env")
)
print(config["VAULT_NAME"])