from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import sqlite3
from datetime import datetime
from health_advisor import generate_health_recommendations
from chatbot import chatbot
import document_analysis
import logging
import speech_recognition as sr


app = Flask(__name__)

# Database initialization
def init_db():
    conn = sqlite3.connect('diabetes_predictions.db')
    c = conn.cursor()
    # predictions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gender TEXT,
            age INTEGER,
            hypertension INTEGER,
            heart_disease INTEGER,
            smoking_history TEXT,
            bmi REAL,
            HbA1c_level REAL,
            blood_glucose_level INTEGER,
            prediction INTEGER,
            probability REAL,
            timestamp DATETIME
        )
    ''')
    # blood sugar readings table
    c.execute('''
        CREATE TABLE IF NOT EXISTS sugar_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fasting INTEGER,
            after_meal INTEGER,
            timestamp DATETIME
        )
    ''')
    # pressure readings table
    c.execute('''
        CREATE TABLE IF NOT EXISTS pressure_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            systolic INTEGER,
            diastolic INTEGER,
            timestamp DATETIME
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database when app starts
init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/tracking')
def tracking():
    return render_template('tracking.html')

@app.route('/chatbot', methods=['GET'])
def chatbot_page():
    return render_template('chatbot.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])
        model = joblib.load('Random-Forest-diabetes.pkl')
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        # Save to database
        conn = sqlite3.connect('diabetes_predictions.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO predictions (
                gender, age, hypertension, heart_disease, 
                smoking_history, bmi, HbA1c_level, blood_glucose_level,
                prediction, probability, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['gender'],
            data['age'],
            data['hypertension'],
            data['heart_disease'],
            data['smoking_history'],
            data['bmi'],
            data['HbA1c_level'],
            data['blood_glucose_level'],
            int(prediction[0]),
            float(max(probability[0])),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        conn.commit()
        conn.close()
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(max(probability[0]))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/save_sugar', methods=['POST'])
def save_sugar():
    try:
        data = request.get_json()
        conn = sqlite3.connect('diabetes_predictions.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO sugar_readings (fasting, after_meal, timestamp)
            VALUES (?, ?, ?)
        ''', (
            data['fasting'],
            data['after_meal'],
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        conn.commit()
        conn.close()
        return jsonify({'message': 'the blood sugar readings have been saved successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/save_pressure', methods=['POST'])
def save_pressure():
    try:
        data = request.get_json()
        conn = sqlite3.connect('diabetes_predictions.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO pressure_readings (systolic, diastolic, timestamp)
            VALUES (?, ?, ?)
        ''', (
            data['systolic'],
            data['diastolic'],
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        conn.commit()
        conn.close()
        return jsonify({'message': 'the blood pressure readings have been saved successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_readings', methods=['GET'])
def get_readings():
    try:
        conn = sqlite3.connect('diabetes_predictions.db')
        c = conn.cursor()
        c.execute('SELECT * FROM sugar_readings ORDER BY timestamp DESC LIMIT 100')
        sugar_readings = [{'id': row[0], 'fasting': row[1], 'after_meal': row[2], 'timestamp': row[3]} for row in c.fetchall()]
        c.execute('SELECT * FROM pressure_readings ORDER BY timestamp DESC LIMIT 100')
        pressure_readings = [{'id': row[0], 'systolic': row[1], 'diastolic': row[2], 'timestamp': row[3]} for row in c.fetchall()]
        conn.close()
        return jsonify({
            'sugar_readings': sugar_readings,
            'pressure_readings': pressure_readings
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_averages', methods=['GET'])
def get_averages():
    try:
        conn = sqlite3.connect('diabetes_predictions.db')
        c = conn.cursor()
        c.execute('''
            SELECT 
                AVG(fasting) as avg_fasting_7,
                AVG(after_meal) as avg_after_meal_7,
                AVG(systolic) as avg_systolic_7,
                AVG(diastolic) as avg_diastolic_7
            FROM (
                SELECT fasting, after_meal, NULL as systolic, NULL as diastolic
                FROM sugar_readings
                WHERE timestamp >= date('now', '-7 days')
                UNION ALL
                SELECT NULL as fasting, NULL as after_meal, systolic, diastolic
                FROM pressure_readings
                WHERE timestamp >= date('now', '-7 days')
            )
        ''')
        week_averages = c.fetchone()
        c.execute('''
            SELECT 
                AVG(fasting) as avg_fasting_30,
                AVG(after_meal) as avg_after_meal_30,
                AVG(systolic) as avg_systolic_30,
                AVG(diastolic) as avg_diastolic_30
            FROM (
                SELECT fasting, after_meal, NULL as systolic, NULL as diastolic
                FROM sugar_readings
                WHERE timestamp >= date('now', '-30 days')
                UNION ALL
                SELECT NULL as fasting, NULL as after_meal, systolic, diastolic
                FROM pressure_readings
                WHERE timestamp >= date('now', '-30 days')
            )
        ''')
        month_averages = c.fetchone()
        conn.close()
        return jsonify({
            'week_averages': {
                'fasting': round(week_averages[0], 1) if week_averages[0] else 0,
                'after_meal': round(week_averages[1], 1) if week_averages[1] else 0,
                'systolic': round(week_averages[2], 1) if week_averages[2] else 0,
                'diastolic': round(week_averages[3], 1) if week_averages[3] else 0
            },
            'month_averages': {
                'fasting': round(month_averages[0], 1) if month_averages[0] else 0,
                'after_meal': round(month_averages[1], 1) if month_averages[1] else 0,
                'systolic': round(month_averages[2], 1) if month_averages[2] else 0,
                'diastolic': round(month_averages[3], 1) if month_averages[3] else 0
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/delete_last_sugar', methods=['POST'])
def delete_last_sugar():
    try:
        conn = sqlite3.connect('diabetes_predictions.db')
        c = conn.cursor()
        c.execute('DELETE FROM sugar_readings WHERE id = (SELECT MAX(id) FROM sugar_readings)')
        conn.commit()
        conn.close()
        return jsonify({'message': 'Last blood sugar reading deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/delete_last_pressure', methods=['POST'])
def delete_last_pressure():
    try:
        conn = sqlite3.connect('diabetes_predictions.db')
        c = conn.cursor()
        c.execute('DELETE FROM pressure_readings WHERE id = (SELECT MAX(id) FROM pressure_readings)')
        conn.commit()
        conn.close()
        return jsonify({'message': 'Last blood pressure reading deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/clear_sugar', methods=['POST'])
def clear_sugar():
    try:
        conn = sqlite3.connect('diabetes_predictions.db')
        c = conn.cursor()
        c.execute('DELETE FROM sugar_readings')
        conn.commit()
        conn.close()
        return jsonify({'message': 'All blood sugar readings cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/clear_pressure', methods=['POST'])
def clear_pressure():
    try:
        conn = sqlite3.connect('diabetes_predictions.db')
        c = conn.cursor()
        c.execute('DELETE FROM pressure_readings')
        conn.commit()
        conn.close()
        return jsonify({'message': 'All blood pressure readings cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_recommendations', methods=['GET'])
def get_recommendations():
    try:
        # Get the language parameter from the request
        language = request.args.get('language', 'arabic')
        # Check if the language is valid
        if language not in ['arabic', 'english']:
            language = 'arabic'
        
        # Generate health recommendations
        recommendations = generate_health_recommendations(language)
        return jsonify({'success': True, 'recommendations': recommendations})
    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}") 
        return jsonify({'success': False, 'error': str(e)}), 500

# Route for chatbot endpoint
@app.route('/chatbot', methods=['POST'])
def handle_chatbot():
    return chatbot()

# Route for chatbot endpoint
@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    from chat_db import Conversation, db
    with db:
        conversations = Conversation.select().order_by(Conversation.last_update.desc())
        return jsonify([{
            'id': conv.conversation_id,
            'start_time': conv.start_time.isoformat(),
            'last_update': conv.last_update.isoformat()
        } for conv in conversations])

@app.route('/api/conversations/<conversation_id>', methods=['GET'])
def get_conversation_messages(conversation_id):
    from chat_db import Conversation, Message, db
    with db:
        try:
            conversation = Conversation.get(Conversation.conversation_id == conversation_id)
            messages = Message.select().where(
                Message.conversation == conversation
            ).order_by(Message.timestamp)
            
            return jsonify({
                'conversation_id': conversation_id,
                'messages': [{
                    'content': msg.content,
                    'is_user': msg.is_user,
                    'timestamp': msg.timestamp.isoformat()
                } for msg in messages]
            })
        except Conversation.DoesNotExist:
            return jsonify({'error': 'Conversation not available'}), 404

@app.route('/delete_conversation', methods=['POST'])
def delete_conversation():
    try:
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        
        if not conversation_id:
            return jsonify({'error': 'Conversation ID is required'}), 400
            
        
        from chat_db import Conversation, Message, db
        with db:
            conversation = Conversation.get_or_none(Conversation.conversation_id == conversation_id)
            if conversation:
                
                Message.delete().where(Message.conversation == conversation).execute()

                conversation.delete_instance()
                return jsonify({'success': True}), 200
            else:
                return jsonify({'error': 'Conversation not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_document', methods=['GET', 'POST'])
def analyze_document():
    if request.method == 'POST':
        try:
            # get the file from the request
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            # analyze the document
            analysis_result = document_analysis.analyze_document(file)
            return jsonify({'analysis': analysis_result})

        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            return jsonify({'error': str(e)}), 500
    else:
        # render the analyze document page
        return render_template('analyze_document.html')

@app.route('/recognize_speech', methods=['POST'])
def recognize_speech():
    try:
        audio_data = request.files['audio_data']
        language = request.form.get('language', 'en-US')
        recognizer = sr.Recognizer()
        audio_content = audio_data.read()
        audio = sr.AudioData(audio_content, audio_data.content_type, audio_data.mimetype)
        text = recognizer.recognize_google(audio, language=language)
        return jsonify({'text': text})
    except sr.UnknownValueError:
        return jsonify({'error': 'Sorry, I did not understand that.'}), 400
    except sr.RequestError as e:
        return jsonify({'error': f'Could not request results; {e}'}), 500




# Helper functions for database queries
def get_sugar_readings():
    conn = sqlite3.connect('diabetes_predictions.db')
    c = conn.cursor()
    c.execute('SELECT fasting, after_meal FROM sugar_readings ORDER BY timestamp DESC LIMIT 1')
    readings = c.fetchone()
    conn.close()
    return {
        'fasting': readings[0] if readings else None,
        'after_meal': readings[1] if readings else None
    }

def get_pressure_averages(period='7'):
    conn = sqlite3.connect('diabetes_predictions.db')
    c = conn.cursor()
    c.execute(f'''
        SELECT AVG(systolic), AVG(diastolic)
        FROM pressure_readings
        WHERE timestamp >= date('now', '-{period} days')
    ''')
    averages = c.fetchone()
    conn.close()
    return {
        'systolic': round(averages[0], 1) if averages[0] else None,
        'diastolic': round(averages[1], 1) if averages[1] else None
    }


if __name__ == '__main__':
    from chat_db import initialize_db
    initialize_db()
    app.run(debug=True)