# Smart Medical System: AI-Powered Health Monitoring and Analysis


![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

![Smart-Medical-System](https://github.com/FerasAlshash/Smart-Medical-System/blob/main/Smart%20Medical%20System.png)

HealthSync is a web application designed to assist users in predicting diabetes risk, tracking health metrics (blood sugar and blood pressure), and analyzing medical reports using AI-powered tools.
Built with Python and Flask, it integrates machine learning, natural language processing (NLP), and optical character recognition (OCR) to provide a comprehensive health management experience.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)
- [Contact](#contact)

## Features
- **Diabetes Prediction**: Predict diabetes risk using a pre-trained Random Forest model based on user-input health data.
- **Health Tracking**: Record and visualize blood sugar and blood pressure readings with historical averages.
- **AI Chatbot**: Interact with a medical assistant chatbot powered by Mistral AI for health-related queries.
- **Document Analysis**: Upload PDF reports or images to extract and analyze medical data with tailored health recommendations.
- **Speech Recognition**: Convert voice input to text for hands-free operation.
- **Multi-Language Support**: Supports English, Arabic, and German for document analysis and chatbot responses.

## Technologies Used
- **Backend**: Python 3.8+, Flask
- **Machine Learning**: scikit-learn (Random Forest), joblib
- **AI/NLP**: LangChain, Mistral AI (Pixtral-12B)
- **Database**: SQLite
- **Image Processing**: OpenCV, PyMuPDF
- **OCR**: OCR.space API
- **Speech Recognition**: SpeechRecognition library
- **Frontend**: HTML, CSS, JavaScript (assumed for templates)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/FerasAlshash/Diabetes-prediction.git


2. **Create a Virtual Environment**
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Dependencies**
pip install -r requirements.txt

*  Note: If requirements.txt is not provided, install the following manually:
*  pip install flask pandas joblib sqlite3 langchain langchain-mistralai requests pillow pymupdf numpy opencv-python speechrecognition

4. **Set Up API Keys**
Mistral AI: Obtain an API key from Mistral AI and replace the placeholder in chatbot.py and document_analysis.py

5. **Initialize the Database**
The application automatically creates an SQLite database (diabetes_predictions.db) on first run.

### Usage
1. **Run the Application 🚀**

*  python app.py

The app will start on http://localhost:5000 in debug mode.

2. **Access Features**

*  Home: Visit http://localhost:5000/ for the main page.

*  Prediction: Go to /prediction to input health data and get diabetes risk.

*  Tracking: Use /tracking to log and view blood sugar and blood pressure.

*  Chatbot: Access /chatbot to interact with the AI assistant.

*  Document Analysis: Navigate to /analyze_document to upload reports.

3. **Example Workflow**

*  Upload a blood test PDF at /analyze_document.

*  The app extracts text, analyzes it, and returns health insights.

*  Ask follow-up questions via the chatbot.

## Project Structure 📂

Smart Medical System/

│

├── app.py                  # Main Flask application

├── chatbot.py              # Chatbot logic with Mistral AI integration

├── document_analysis.py    # Document processing and analysis

├── chat_db.py              # Database models for chatbot conversations (assumed)

├── health_advisor.py       # Health recommendation generator (assumed)

├── templates/              # HTML templates (e.g., index.html, prediction.html)

│   ├── index.html

│   ├── prediction.html

│   ├── tracking.html

│   ├── chatbot.html

│   └── analyze_document.html

├── static/                     # CSS, JS, and other static files (assumed)

├── Random-Forest-diabetes.pkl  # Pre-trained diabetes prediction model

├── diabetes_predictions.db     # SQLite database (generated)

└── README.md                   # Project documentation

### License 📜

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact Information 📞

For any questions or feedback, feel free to reach out:

- **Email**: [ferasalshash@gmail.com](mailto:ferasalshash@gmail.com)  
- **GitHub**: [FerasAlshash](https://github.com/FerasAlshash)  
- **LinkedIn**: [Feras Alshash](https://www.linkedin.com/in/feras-alshash-bb3106a9/)  


