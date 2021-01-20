# um-lime
LIME project for the Machine Learning course

# Tutorial

You will need a Docker and docker-compose on your machine to run the tutorial.

## Installation
1. Clone the repository
2. Checkout the segmentation branch `git checkout segmentation`
2. Execute `docker build -t agh/um-lime .`. (Building an image maight take a few minutes)
3. Run the container with `docker-compose up`.

## Clean up
1. Remove the container with `docker-compose down -v`.
2. Remove the image with `docker rmi agh/um-lime`.
