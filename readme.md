# PHOTONAI WIZARD

This is a local installation of the PHOTONAI Wizard, an educational machine learning code generator for the Python package [PHOTONAI](www.photon-ai.com). A publicly available running version can be found at [wizard.photon-ai.com](wizard.photon-ai.com).

The PHOTONAI Wizard guides you through a series of steps involved in the definition of a machine learning analysis, such as cross-validation, data preprocessing, algorithm selection or performance metrics.

## SETUP

The installation of the PHOTONAI Wizard should be simple and straightforward. The only necessary requirement is a docker installation. Everything else will be handled within the docker container.



Requirements:

- Git

* Docker

  

Steps:

1. in the terminal, clone the PHOTONAI Wizard repository into a folder of your choice using

   `git clone https://github.com/wwu-mmll/photonai_wizard.git`

2.   cd into the repository and run 

   `sudo docker-compose build`

   `sudo docker-compose up`

3. go to the following page in your web browser

   `localhost:80`

   the wizard should now be ready to use