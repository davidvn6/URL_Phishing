# URL Phishing Detector

A machine learning application that detects phishing URLs using logistic regression and supervised learning classification and classifies them as either safe or malicious. 

## Executable Folder
This folder contains a precompiled executable version of the application.

**To run the application using the executable:**
- Navigate to the `Executable Folder`
- Double click the CPSC481Proj Application.ex file to launch the application
- You can then click the development server URL in the terminal: `http://127.0.0.1:5000`
- Enter a URL to check if it's malicious or safe.
```
   https://www.facebook.com/ (Output will deem as safe)

   https://paypa1-secure-login.com/verify/account 
   (Output will deem as malcious)
```
- No Python installation is required for this method

If the executable does not run, use the **Code Folder** method below.

## Code Folder
This folder contains the source code for the application.

Download all files in this folder.

Use this method if:
- The executable file does not work
- You want to view or modify the source code

Follow all instructions below.
## Requirements
- Python 3.8 or higher
- pip (included with Python)

## Installation
Ensure you have python installed on your machine.
Download and install Python from:
```bash
https://www.python.org/downloads/
```

Install the packages with:
```bash
pip install -r requirements.txt
```

- Python 3.8 or higher
- Flask
- pandas
- scikit-learn
- NumPy
- SciPy
- joblib


## Usage

1. Run the application:
```bash
   python app.py
```

2. Open the application in one of the following ways:
- Click the development server URL in the terminal: `http://127.0.0.1:5000`
- Or open your browser and navigate to: `http://localhost:5000`



3. Enter a URL to check if it's malicious or safe.
```
   https://www.facebook.com/ (Output will deem as safe)

   https://paypa1-secure-login.com/verify/account 
   (Output will deem as malcious)
```

