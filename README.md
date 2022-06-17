# Project2
Disaster Response Pipeline Project

Michael Bono


This project works through the ETL to modeling process in classifying messages based on their 


**What's Included In This Repository **

- data/process_data.py (Python file containing code to clean and prepare data for modeling)
- data/disaster_messages.csv (source file containing message data)
- data/disaster_categories.csv (source file containing categories of disaster for classification)
- models/train_classifier.py (Python file containing code to train and test a classification model)
- app/run.py (Python file containing code to create the app interface to review data and classification model)
- app/templates (folder containing the HTML code to create the webpage for the app)
- This README.md file (a brief description and summary of the project and libraries used)

**Libraries Used**

- numpy
- pandas
- math
- matplotlib
- sklearn
- nltk
- sqlalchemy
- warnings
- re
- pickle

**Methodology**

This analysis followed the CRISP-DM method to gather, assess, clean, analyze, model, and visualize the data. This analysis then created an ETL and modeling pipeline to prepare the data and create the classification model, as well as a Flask app to review results.

**Instructions:**
1. Run the following commands in the project's root directory to set up your database and model.
    a. To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    b. To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
2. Run the following command in the app's directory to run your web app. python run.py
3. Go to http://0.0.0.0:3001/ Or Go to http://localhost:3001/

**Thanks**

Thank you for checking out this project! I hope you learned something interesting.
