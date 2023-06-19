# Deployment

**IMPORTANT:** Make sure to have an Azure File Share created, for example `search-qna-data`, as it will be mounted onto the containers as Docker Volumes onto the `/code/data` path. This ensures that multiple containers have the same access to persistent data and models stored within.

---

This document describes the deployment to Azure in two ways:

1. Azure Container Instances
2. Azure App Service

The built containers are configured to have [SSH access for App Service](https://learn.microsoft.com/en-us/azure/app-service/configure-custom-container?tabs=debian&pivots=container-linux#enable-ssh).

GitHub Actions uses the `dev.dockerfile` for building images from the `development` branch and pushing on the Dev Environment on Azure Container Registry, and the `prod.dockerfile` for the same on `master` branch for the Production environment.


With 1 FastAPI Worker in `app.py`, the application needs a little over 3 GB of memory to load all the models properly. Failing to assign enough memory (or enough startup time in case of App Service) will result in the deployment not working. Need to check the memory and startup time requirements in case of 2 or more workers.

---

## Deployment to Azure Container Instances

Here, the first step would be remove the `ENTRYPOINT ["init.sh"]` line from the Dockerfiles. This is because we will provide a custom startup command while creating containers. Build the image and push to the Azure Container Registry (if needed) as follows:


### Building Images

We look at two different ways, use whichever is more comfortable.

#### With Docker

You can use the [Docker Integration with ACI](https://docs.docker.com/cloud/aci-integration/).

Build the image in the standard way locally, for example `docker build -t semanticsearchqna:latest .` and then push onto ACR as follows:

- Login to the `fabricacr123` registry. You can get the login URI and credentials from Access keys on the `fabricacr123` page on Azure Portal.

  ```bash
  docker login fabricacr123.azurecr.io
  ```

- Tag the image

  ```bash
  docker tag semanticsearchqna:latest fabricacr123.azurecr.io/semanticsearchqna:latest
  ```

  The `semanticsearchqna:latest` is only an example image name.

- Push the image onto ACR.

  ```bash
  docker push fabricacr123.azurecr.io/semanticsearchqna:latest
  ```

---

#### With Azure CLI

After you have logged into Azure with the CLI,

```bash
az acr build --image <image_name> --registry <container_name> --file Dockerfile .
```

This will build the image with Azure's resources, so Docker setup is not required at all in this case.

---

### Creating Relevant Container Instances

We use the Azure CLI to create container instances, specifically the [az container create](https://learn.microsoft.com/en-us/cli/azure/container?view=azure-cli-latest) command. Simply creating from Azure Portal will not work as the Portal does not (at the time of writing) give the option to mount a File Share as a Docker Volume.

There are four containers that may be created. Make sure to use the proper image name. Remember to also substitute `ACR_PASSWD` (ACR Password) and `AFS_KEY` (Storage Account's Key). They may be found from the Portal, or in the `.env` file.

Among the commands for these containers, only the Startup Command (`--command`) and Restart Policy (`--restart-policy`) change. For the application, however, DNS Label and Ports are also specified.

1. **Create Models**

This one is only required to be run once, when just starting out fresh in a new File Share. This will create the relevant directories and download and store the appropriate mdoels.

This container can be deleted later on from the Portal.

```bash
az container create \
--resource-group FABRIC \
--name search-qna-create-resources \
--image fabricacr123.azurecr.io/semanticsearchqna:latest \
--registry-username fabricacr123 \
--registry-password $ACR_PASSWD \
--azure-file-volume-share-name search-qna-data \
--azure-file-volume-account-name fabricaccount \
--azure-file-volume-account-key $AFS_KEY \
--azure-file-volume-mount-path /code/data \
--cpu 2 --memory 4 \
--location centralindia \
--command "python3 create_models.py" \
--restart-policy OnFailure
```


2. **Fetch Data**

This container fetches data from CosmosDB, computes and updates embeddings. This can be scheduled to run everyday. One way to accomplish that is by using Azure Logic Apps.

```bash
az container create \
--resource-group FABRIC \
--name search-qna-fetch-data \
--image fabricacr123.azurecr.io/semanticsearchqna:latest \
--registry-username fabricacr123 \
--registry-password $ACR_PASSWD \
--azure-file-volume-share-name search-qna-data \
--azure-file-volume-account-name fabricaccount \
--azure-file-volume-account-key $AFS_KEY \
--azure-file-volume-mount-path /code/data \
--cpu 2 --memory 4 \
--location centralindia \
--command "python3 fetch_from_DB.py" \
--restart-policy Never
```

3. **Run App**

The application should always be running, hence the Restart Policy of _Always_.

```bash
az container create \
--resource-group FABRIC \
--name search-qna-app \
--image fabricacr123.azurecr.io/semanticsearchqna:latest \
--registry-username fabricacr123 \
--registry-password $ACR_PASSWD \
--azure-file-volume-share-name search-qna-data \
--azure-file-volume-account-name fabricaccount \
--azure-file-volume-account-key $AFS_KEY \
--azure-file-volume-mount-path /code/data \
--ports 80 443 8000 \
--dns-name-label semantic-search-qna \
--cpu 2 --memory 4 \
--location centralindia \
--command "python3 app.py" \
--restart-policy Always
```

4. **Remove Older Data**

This container removes data older than 30 days. This can be scheduled to run everyday, or every week, as feasible. One way to accomplish that is by using Azure Logic Apps.

```bash
az container create \
--resource-group FABRIC \
--name search-qna-fetch-data \
--image fabricacr123.azurecr.io/semanticsearchqna:latest \
--registry-username fabricacr123 \
--registry-password $ACR_PASSWD \
--azure-file-volume-share-name search-qna-data \
--azure-file-volume-account-name fabricaccount \
--azure-file-volume-account-key $AFS_KEY \
--azure-file-volume-mount-path /code/data \
--cpu 2 --memory 4 \
--location centralindia \
--command "python3 remove_older_docs.py" \
--restart-policy OnFailure
```

---

### Troubleshooting Container Instances

You can check the logs of a Container Instance using the [`az container logs`](https://learn.microsoft.com/en-us/cli/azure/container?view=azure-cli-latest#az-container-logs) command.

```bash
az container logs --resource-group FABRIC --name ContainerInstanceName
```

## Deployment to Azure App Services

In this case, do NOT remove the `ENTRYPOINT ["init.sh"]` from the Dockerfiles. They are configured to allow SSH access to the container from App Service. 

### Creating the App through Portal

The App Service application has to be created through the Azure Portal.

- Go to App Service -> Create App.
- Select the Publish source as Docker.
- Make sure to select a Pricing Plan resource with 4 GB memory or higher.
    - If the deployment fails, try to select another region, like East US.
- Give an appropriate Name, like `search-qna-app` for example.
- In the Docker tab, select the Image Source as Azure Container Registry and select the appropriate image.
- Create the App Service application.

### Mounting File Share

Now, we need to mount the File Share onto this application.

Resources:
- [Documentation Page](https://learn.microsoft.com/en-us/azure/app-service/configure-connect-to-azure-storage?tabs=portal&pivots=container-linux)
- [Video](https://www.youtube.com/watch?v=OJkvpWYr57Y)

Steps:
- Go to the App Service app in Portal
- Go to Configuration -> Path Mappings
- Make sure to set a name for the mount with all small letters and no spaces
- Select the File Share and set the mount path to `/code/data`. Click on Save.

### Increasing Start Time Limit

Another important thing to note is that the Azure App Service waits for the application to start for 240 seconds by default, after which it will force a restart. The current Semantic Search + QnA application takes around 700-800 seconds to start (with 1 worker in FastAPI). So, the default wait time for startup has to be increased.

Refer to the [Documentation Page](https://learn.microsoft.com/en-us/archive/blogs/waws/things-you-should-know-web-apps-and-linux#if-your-container-takes-a-long-time-to-start-increase-the-start-time-limitapplies-to-web-app-for-containers) and set an App Setting called `WEBSITES_CONTAINER_START_TIME_LIMIT` to `900` or `1200` (seconds), whatever seems feasible.

Both the File Share mount and the default start time limit increase has to be done in order for the application to work successfully, otherwise it will fail to run and keep restarting (and failing) in a perpetual loop.

Scheduling the fetch data and remove older data programs is yet to be figured out for the App Service.

### Troubleshooting App Services

You can go to the App through the Portal, and check the "Diagnose and solve problems" section, from where you can access the Container logs and Application logs. Reading through them might provide some helpful information in case something goes wrong.

---

## Acknowledgements

Special Thanks to Anoushka Rao and Padmanabh Manikandan, who provided a lot of guidance on many of these, especially helping with Troubleshooting when things weren't working as expected.