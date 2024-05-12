# How to run Docker container
## Pull image and run in CLI mode
### pull 
`docker pull huongntt/tnmt-api-server`
### run
`docker run -p 5000:<your_expected_port> huongntt/tnmt-api-server`

## Pull image and run in Docker Desktop 
### pull 
1. ctrl+K: search for 'huongntt/tnmt-api-server'
2. pull image 
### run
1. Images tab: action RUN
2. Fill the necessary parameters in Optional Settings
    Container name (ex: `tnmt-api-server-container`)
    Host port <your_expected_port>

## Result 
{
  "Application": "TNMT api service"
}


# How to build an image 
### Building and running your application

When you're ready, start your application by running:
`docker compose up --build`.

Your application will be available at http://localhost:5000.

### Deploying your application to the cloud

First, build your image, e.g.: `docker build -t myapp .`.
If your cloud uses a different CPU architecture than your development
machine (e.g., you are on a Mac M1 and your cloud provider is amd64),
you'll want to build the image for that platform, e.g.:
`docker build --platform=linux/amd64 -t myapp .`.

Then, push it to your registry, e.g. `docker push myregistry.com/myapp`.

Consult Docker's [getting started](https://docs.docker.com/go/get-started-sharing/)
docs for more detail on building and pushing.

### References
* [Docker's Python guide](https://docs.docker.com/language/python/)

# Container architecture ?? 
## OS
`docker exec <container_id> uname -a`
Linux f87305c6d412 5.15.146.1-microsoft-standard-WSL2 #1 SMP Thu Jan 11 04:09:03 UTC 2024 x86_64 GNU/Linux

## OS
`docker exec <container_id> env`
PYTHON_VERSION=3.11.9
PYTHON_PIP_VERSION=24.0
PYTHON_SETUPTOOLS_VERSION=65.5.1