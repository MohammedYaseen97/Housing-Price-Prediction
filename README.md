# PES Real Estate : Housing Price Prediction		

We present various important features to use while predicting housing prices with good accuracy. While using features in a regression model some feature engineering is required for better prediction. Often a set of features (multiple regressions) or polynomial regression (applying a various set of powers in the features) is used for making better model fit. While in Neural Networks, solely the difference between the predicted and existing values determines the change in weights, hence determining the best fit.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them
```
Hardware Requirements
• Processor: 2GHz or faster processor
• RAM: 2GB(64bit)
• Storage: 5GB of available hard disk space
• Consistent internet connection having speed of 512Kbps at a bare minimum.
• Other general hardwares such as a mouse and keyboard for inputs and a monitor for display.
```
```
Software Requirements
• Operating system: Linux or Windows
• Programming languages: Python
• Anaconda (Open Source Python Distribution) link : https://www.anaconda.com/distribution/
```

### Running

Navigate to the project root directory in the anaconda prompt shell

Install all the required libraries. Type the following command : 

```
pip install -r requirements.txt
```

Run the script :  

```
python app/start.py
```

Open a web browser and navigate to localhost:5000

Since the scripts are too heavy and the training takes a long time, you will have to wait a good couple minutes to see the results. What you will see in front of you is the website of PES Real Estate. In the box of "Get going!", type in your lot area, no. of bedrooms and year built, and click "Predict". The page will reload (for a long time) and you will have your answer in the "Cost" textbox.
