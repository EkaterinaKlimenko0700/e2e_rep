# FINAL TASK
The main objective of this assignment was to create a machine learning model to predict real estate prices. And then it was necessary to save this model in a web service, test it and Dockerize the application

# Something about source data
```
The source data for this Lab comes from Yandex.Realty classified, accessible through https://realty.yandex.ru. It contains comprehensive real estate listings for apartments located in St. Petersburg and the Leningrad Oblast. The dataset spans from 2016 to the middle of August 2018. 
```

# Data processing
- First, I downloaded the libraries (pandas, sklearn) and data

- Loaded data from the file `spb.real.estate.archive.2018.tsv` using the `\t` delimiter

- We look through the columns to understand the structure of the data proposed for analysis.

Filtered the data by `offer_type` = 2

- Filtered data by location `St. Petersburg`

- Selected columns for analysis - 'last_price', 'floor',  'rooms', 'area', 'renovation'

- Selected floors from 1 to 20, took as a basis apartments with at least 1 and no more than 3 rooms with an area from 20 to 90 square meters maximum

Divided the data into X ('floor',  'rooms', 'area', 'renovation') and Y ('last_price'), as well as into training and test samples

# Model Hyperparameters

`parameters = {'n_estimators':[i for i in range(10,110,10)], 'max_depth':[1,2,3,4, 5,6]}` 

- Applied the RandomForest model and selected hyperparameters using GridSearchCV and found the best

- Trained the final model with the best parameters

- Used the trained model to predict values ​​on the test data set.

- Calculate the ambient error for predictions.

---

- Serialized the model using joblib

- Created a Flask web service

```
from flask import Flask, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load ("f.joblib")
sc_x = joblib.load ("x.joblib")
sc_y = joblib.load ("y.joblib")

@app.route('/api/ml/predict/price', methods=['GET'])
def predict_price():
    args = request.args

    floor = args.get('floor', default = -1, type=int)
    rooms = args.get('rooms', default = -1, type=int)
    area = args.get ('area', default = -1, type=float)
    renovation = args.get('renovation', default = -1, type=int)
    
    x = np.array([floor, rooms, area, renovation]).reshape (1,-1)
    x = sc_x.transform(x)

    result = model.predict(x)
    result = sc_y.inverse_transform(result.reshape (1, -1))

    return str(result[0][0])

if __name__ == '__main__':
    app.run(host= '0.0.0.0',debug=True)

```

```

- Then I loaded a previously trained model from rf.joblib.
- There was a prediction using the model.

```

## Installation Steps
1. Clone the repository:
  https://github.com/EkaterinaKlimenko0700/e2e_rep

2. Create a virtual environment:
   bash
   python3 -m venv venv
   source venv/bin/activate  

3. Install the dependencies: pip install -r requirements.txt
   
### Running the App
1. Start the Flask app:
   python3 app.py

2. The app will be available at `http://0.0.0.0:5000`.

# Connecting 
```
I connected to the host 158.160.9.33 under the user st081184 using an SSH key and now logged into the terminal of the remote machine. The next task was to install Docker since the docker command is not installed on the remote machine.

```
# Docker

### Dockerfile

```
FROM ubuntu:20.04
RUN apt-get update -y
COPY . /opt/gsom_predictor
WORKDIR /opt/gsom_predictor
RUN apt install -y python3-pip
RUN pip3 install -r requirements.txt
CMD python3 app.py

```

The official Python version 3 image is used, which already contains Python 3 installed and basic tools for working with it. I set the working directory, copied the requirements.txt file from the current directory to the working directory. Installed the dependencies listed in the requirements.txt file using pip and the command that will be executed when the container starts - run app.py using Python.

### Loading

Loaded tables and data, created a virtual Python environment, installed docker. Launched the container via sudo (hello-world)

Next, the process of creating a Docker image using docker build began. After success, I launched the Docker container using docker run, opening the necessary ports.

`sudo docker run -d -p 5000:5000 --rm --name ml_app ml_app`


# Connecting to a virtual machine

ssh <user_name>@<VM_public_IP-address>
