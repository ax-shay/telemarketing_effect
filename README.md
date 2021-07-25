# UDCDSA Captsone Project: Predicting Effect of Bank Telemarketing (Term Deposit Sale)
## Executive Summary
### Goal: 
To predict if the banking clients will subscribe to a term deposit
### Overview:
This research project focuses on targeting through telemarketing phone calls to sell long-term deposits. Within a campaign, the human agents execute phone calls to a list of clients to sell the deposit (outbound) or, if meanwhile the client calls the contact-center for any other reason, he is asked to subscribe the deposit (inbound). Thus, the result is a binary unsuccessful or successful contact.<br>
### Team Members:
Akshay P. Shembekar; Courtney Golding; Jonathan Littleton; Komal Handa; Sambhavi Parajuli


## Files

### Input:
Dataset Link: https://archive.ics.uci.edu/ml/datasets/bank+marketing

There is one input dataset:
1) **bank-additional-full.csv**:  
   - It has 41188 x 20 inputs, ordered by date (from May 2008 to November 2010)

The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

### Code:
BitBucket Repo: https://github.com/ax-shay/telemarketing_effect

1) **requirements.txt**:
    - All python import dependencies. 

2) **tele-marketing.py**:
    - Main application file which houses the Streamlit application code. 

3) **home.py**:
    - Executive Summary of Project. 

4) **eda.py**:
    - Findings for Exploratory Data Analysis (EDA). 

5) **model.py**:
    - A head-to-head model comparison for 3 distinct models. 

6) **predictions.py**:
    - The best model amongst the 3 used for predicting the subscriber-ship of a client. 

7) **Misc**:
    - Procfile: For running app on Heroku server
    - setup.sh: For running app on Heroku server
    - defSessionState.py: Manage paging within the app. 

### Presentation:
The "Presentation" folder contains .pptx & .pdf version of the final presentation for this project.

### AutoML:
Check-out the "TPOT" folder which contains code used to run AutoML using TPOT classifier.
This also contains profiling of data done using pandas_profile before clean-up and after clean-up of data.

## Application:
Link: https://telemarketing-effect.herokuapp.com

-- END
