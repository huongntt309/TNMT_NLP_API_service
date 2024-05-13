# TNMT API Service

## Overview
The TNMT API Service is designed to provide various functionalities for classifying and summarizing Natural Resources and Environment news articles. It offers endpoints for classifying articles, summarizing content, and combining both classification and summarization.

## Installation and Setup
To run the TNMT API Service, you should using Docker

1. Ensure you have Docker installed on your system.
    Or download at https://www.docker.com/products/docker-desktop/
2. Pull the TNMT API Docker image from Docker Hub:
    ```
    docker pull huongntt/tnmt-api-service-server:v2-ui
    ```
3. Run the Docker container:
    ```
    docker run -p 5000:5000 huongntt/tnmt-api-service-server:v2-ui
    ```
4. The API service will be accessible at `http://localhost:5000`.


## Configuration
The API service can be configured through environment variables. You can customize settings such as port number, logging level, and model paths by setting the corresponding environment variables.

## API Usage 
### Option 1: UI Web 
To access the UI web interface, navigate to http://localhost:5000/. Once there, your request to classify will be sent using a POST request to **/sum-cls** endpoint on the server.

## Option 2: Using Postman
1. Create a **POST request** in Postman (or any other application with similar functionality).
2. Set the request body to JSON data from **app/bow_folder/21-test-cases.json** (or any other JSON data you have).
3. Send the **POST request** to **http://localhost:5000/sum-cls**.

## Technology used
1. "VietAI/vit5-base": ViT5, Pre-trained Text-to-Text Transformer for Vietnamese Language Generation.
2. "vinai/bartpho-syllable": BARTpho, Pre-trained Sequence-to-Sequence Models for Vietnamese.
3. Flask API: A lightweight web framework for Python, offering simplicity and flexibility for building RESTful APIs.

## Contact
For any inquiries or support, please contact [huonguet8@gmail.com](mailto:huonguet8@gmail.com).
