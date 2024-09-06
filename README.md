# Few-Shot-Learning-Bad-Impl-Test

## Description

Few-Shot-Learning-Bad-Impl-Test is a project that demonstrates an implementation of few-shot learning. 

![Alt text](assets/context.png?raw=true "Context Image")

## Prerequisites

Ensure you have Docker installed on your system. If not, you can download and install Docker from [here](https://www.docker.com/products/docker-desktop).

## Setup

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/yourusername/Few-Shot-Learning-Bad-Impl-Test.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Few-Shot-Learning-Bad-Impl-Test
    ```
3. Ensure you have a `requirements.txt` file with the necessary dependencies.

## Running the Project

You can run the project using Docker. The following command will set up the Docker container, install the dependencies, and execute the main script.

```bash
docker run --rm -v "$PWD":/usr/src/app -w /usr/src/app python:3.9 /bin/bash -c "pip install -r requirements.txt && python main.py"
