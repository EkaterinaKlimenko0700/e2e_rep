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

Filtered the data by `offer_type` = 1

- Created a new column `price_m` (price per square meter), which is equal to `last_price` divided by `area`

- Filtered data by location `St. Petersburg`

- Selected columns for analysis

- Afterwards, the data was filtered by area and price (I left lines where the area of ​​the apartment is in the range from 20 to 200 square meters, the price per square meter is in the range from 70,000 to 300,000, I removed `studio`, floors up to 25, buildings with more than records)

Divided the data into X ( `last_day_exposition`) and Y (price_m'), as well as into training and test samples

# Model Hyperparameters

`parameters = {'n_estimators':[i for i in range(10,110,10)], 'max_depth':[1,2,3,4, 5,6]}` 

- Applied the RandomForest model and selected hyperparameters using GridSearchCV and found the best

-Trained the final model with the best parameters

- Used the trained model to predict values ​​on the test data set.

- Calculate the ambient error for predictions.

---

## Correlation heatmap

![Here is relations between area and last_price](.\3.jpg) 

## Box plot of price per meter

![Box plot of price per meter](.\4.jpg) 

## Box plot and violin plot of price per meter

![Box plot and violin plot of price per meter](.\5.jpg) 


- Serialized the model using joblib

- Created a Flask web service

```
from flask import Flask, request, jsonify
import pandas as pd
import joblib
app = Flask(__name__)

list_fields_inout = ['offer_id', 'first_day_exposition', 'last_day_exposition',
       'floor', 'open_plan', 'rooms', 'studio', 'area', 'kitchen_area',
       'living_area', 'agent_fee', 'renovation', 'offer_type', 'category_type',
       'unified_address', 'building_id']

list_filds_for_analis = ['floor', 'rooms', 'area']

```

- list_fields_inout contains a list of all fields that are expected in the input
- list_filds_for_analis contains a list of fields that will be used for prediction (floor, rooms, area)

```

@app.route('/api/ml/predict/price', methods=['GET', 'POST'])
def predict_price():
    content = request.json
    
    list_miss_fields = []

    for item in list_fields_inout:
        if item not in content:
            list_miss_fields.append(item)

    if len(list_miss_fields)>0:
        return jsonify({"success":0, "content":content, "error":"miss", "fields":list_miss_fields}),500

```

The presence of each field from list_fields_inout in the input data is checked:
        - If a field is missing, it is added to list_miss_fields.
        - If list_miss_fields is not empty, a JSON response is returned with an error message, status 500, and a list of missing fields.

 ```
    df = pd.json_normalize(content)
    df = df[list_filds_for_analis]
    
    model = joblib.load("rf.joblib")

    result = model.predict(df)

    return jsonify({"success":1, "content":content, "result":result}), 200

```

- Next I normalized the data
- Selected only the fields necessary for prediction
- Loaded a previously trained model from rf.joblib.
- There was a prediction using the model.

```

if __name__ == '__main__':
    app.run(host= '0.0.0.0',debug=True)
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
   python app.py

2. The app will be available at `http://0.0.0.0:5000`.

# Connecting 
```
I connected to the host 158.160.85.132 under the user st081184 using an SSH key and now logged into the terminal of the remote machine. The next task was to install Docker since the docker command is not installed on the remote machine.

```
# Docker

### Dockerfile

```
FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./app.py" ]

```

The official Python version 3 image is used, which already contains Python 3 installed and basic tools for working with it. I set the working directory, copied the requirements.txt file from the current directory to the working directory. Installed the dependencies listed in the requirements.txt file using pip and the command that will be executed when the container starts - run app.py using Python.

### Loading

Loaded tables and data, created a virtual Python environment, installed docker. Launched the container via sudo (hello-world)

Next, the process of creating a Docker image using docker build began. After success, I launched the Docker container using docker run, opening the necessary ports.

`sudo docker run -d -p 5000:5000 --rm --name ml_app ml_app`

# Postman

![Working Postman with predictions](.\2.jpg) 

# Connecting to a virtual machine

ssh <user_name>@<VM_public_IP-address>