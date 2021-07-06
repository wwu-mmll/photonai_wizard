# PHOTONAI WIZARD

Towards Educational Machine Learning Code Generators

## SETUP

The following is a guide to setting up the wizard **locally**. 

Requirements:

* Docker
* MongoDB

Steps:

1. Setup a MongoDB instance. 
2. Set up wizard
> go to app/wizard-config.py. Define your specific MongoDB instance and other settings. Please follow the specific instructions
> in the config file. Then, go To app/main.py and load your local config.
3. Fill the MongoDB
> go to app/model/db_fill/ - first start element_definition.py, than default_pipelines.py. In some cases the default_pipelines.py
> throws an error. Please drop the hole photon_wizard database (only in your local MongoDB) and repeat Step 3.
4. Build the PHOTONAI Wizard docker
> go to the project root and run 'sudo docker-compose build' and 'sudo docker-compose up' from the terminal
> the PHOTONAI Wizard should now be available in the web browser at 'localhost:7276' 
