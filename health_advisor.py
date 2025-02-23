from langchain_mistralai.chat_models import ChatMistralAI
from langchain.prompts.chat import ChatPromptTemplate
import sqlite3

def get_patient_data():
    """Retrieve patient data for generating health recommendations"""
    conn = sqlite3.connect('diabetes_predictions.db')
    cursor = conn.cursor()
    
    #Retrieve sugar averages for the last week
    cursor.execute("""
        SELECT AVG(fasting) as avg_fasting, AVG(after_meal) as avg_after_meal
        FROM sugar_readings
        WHERE timestamp >= date('now', '-7 days')
    """)
    sugar_weekly = cursor.fetchone()
    
    #Retrieve pressure averages for the last week
    cursor.execute("""
        SELECT AVG(systolic) as avg_systolic, AVG(diastolic) as avg_diastolic
        FROM pressure_readings
        WHERE timestamp >= date('now', '-7 days')
    """)
    pressure_weekly = cursor.fetchone()
    
    # Retrieve the last prediction made
    cursor.execute("""
        SELECT gender, age, hypertension, heart_disease, smoking_history, bmi, 
               HbA1c_level, blood_glucose_level, prediction
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT 1
    """)
    last_prediction = cursor.fetchone()
    
    conn.close()
    
    return {
        'sugar_weekly': sugar_weekly,
        'pressure_weekly': pressure_weekly,
        'last_prediction': last_prediction
    }

def generate_health_recommendations(language='arabic'):
    """Generating health recommendations using the LLM model"""
    # Retrieve patient data
    patient_data = get_patient_data()
    
    # Retrieve the recommendation model
    recommendation_model = ChatMistralAI(
        mistral_api_key="22MIvQ0KgV5u0ZvxZdecv5wmwAwWhMPZ",
        model_name="pixtral-12b-2409"
    )
    
    # initialize the template
    template = """أنت مستشار طبي متخصص في مرض السكري وضغط الدم. قم بتحليل البيانات التالية وتقديم تقرير صحي شامل.

تعليمات مهمة:
1. استخدم اللغة {language} فقط، لا تخلط بين اللغات
2. لا تستخدم أي رموز خاصة مثل #، *، -، _ 
3. استخدم النقطة (•) في بداية كل سطر جديد
4. اكتب النص بشكل مباشر بدون أي تنسيق خاص
5. لا تكرر الكلمات



متوسط قراءات السكر في الدم (آخر 7 أيام)
• مستوى السكر صائم: {avg_fasting} ملغم/ديسيلتر
• مستوى السكر بعد الأكل: {avg_after_meal} ملغم/ديسيلتر

متوسط قراءات ضغط الدم (آخر 7 أيام)
• الضغط الانقباضي: {avg_systolic} ملم زئبق
• الضغط الانبساطي: {avg_diastolic} ملم زئبق

قم بتقديم تقرير يتضمن:

تقرير صحي شامل

تحليل الحالة الصحية
• تقييم مستويات السكر في الدم مقارنة بالمعدلات الطبيعية
• تقييم قراءات ضغط الدم مقارنة بالمعدلات الطبيعية
• تحليل عوامل الخطر المحتملة

الخطة الغذائية المخصصة
• توصيات محددة للوجبات الرئيسية والخفيفة
• قائمة بالأطعمة المفيدة والأطعمة التي يجب تجنبها
• نصائح حول توقيت الوجبات وكمياتها

برنامج النشاط البدني
• أنواع التمارين المناسبة للحالة
• المدة والتكرار الموصى بهما
• احتياطات السلامة أثناء ممارسة الرياضة

علامات تحذيرية تستدعي استشارة الطبيب
• مؤشرات خطيرة في مستويات السكر - مستوى السكر الصائم فوق 250 ملغم/ديسيلتر - مستوى السكر بعد الأكل فوق 200 ملغم/ديسيلتر

• مؤشرات خطيرة في ضغط الدم - ضغط الانقباضي فوق 180 ملم زئبق - ضغط الانبساطي فوق 110 ملم زئبق

• أعراض تستدعي العناية الطبية الفورية:
* الدوخة الشديدة
* الصداع المستمر
* تشوش الرؤية
* خفقان القلب السريع
* التعرق البارد
* الإغماء أو الدوار الشديد

نصائح عامة للحياة اليومية
• إرشادات لتحسين نمط الحياة
• طرق التعامل مع الضغوط النفسية
• أهمية المتابعة الدورية مع الطبيب"""

    # prepare the variables for the template
    sugar_weekly = patient_data['sugar_weekly']
    pressure_weekly = patient_data['pressure_weekly']
    last_prediction = patient_data['last_prediction']
    
    if all(data is not None for data in [sugar_weekly, pressure_weekly, last_prediction]):
        prompt = ChatPromptTemplate.from_template(template)
        
        # Preparing variables for the template
        formatted_message = prompt.format_messages(
            language="العربية" if language == 'arabic' else "English",
            avg_fasting=sugar_weekly[0],
            avg_after_meal=sugar_weekly[1],
            avg_systolic=pressure_weekly[0],
            avg_diastolic=pressure_weekly[1],
            gender=last_prediction[0],
            age=last_prediction[1],
            hypertension=last_prediction[2],
            heart_disease=last_prediction[3],
            smoking=last_prediction[4],
            bmi=last_prediction[5],
            hba1c=last_prediction[6]
        )
        
        # Get recommendations from the model
        response = recommendation_model.invoke(formatted_message)
        return response.content
    else:
        return "Sorry, there is not enough data to make recommendations. Please make sure you have recent blood sugar and blood pressure readings."
