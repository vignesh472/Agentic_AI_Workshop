# Smart Health Assistant 🧑‍⚕️

This is a **Streamlit-based AI-powered Smart Health Assistant** that leverages **Gemini 1.5 Flash** and **AutoGen Agents** to provide personalized health plans. The assistant takes basic health inputs and generates:

- **BMI Analysis**
- **Health Recommendations**
- **Custom Meal Plans**
- **Workout Schedules**

## 🚀 Features

- BMI Calculation using height and weight
- Categorization (underweight, normal, overweight, obese)
- Health advice based on BMI
- Meal plans according to dietary preferences
- Workout schedules based on age, gender, and health goals
- Downloadable personalized health plan

## 🚪 Requirements

- Python 3.8+
- Streamlit
- google-generativeai
- pyautogen (AutoGen)

Install dependencies:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```txt
streamlit==1.36.0
google-generativeai==0.4.1
pyautogen==0.2.28
```

## 🔧 Setup Instructions

1. **Get a Gemini API key** from: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. **Run the app**:

```bash
streamlit run app.py
```

3. **Fill in details** in the sidebar and form:

   - Gemini API key
   - Weight, height, age
   - Gender and dietary preference

4. Click **"Generate Health Plan"**
5. View results and download your custom plan

## 🧑‍💡 Agents Used

- `BMI_Agent`: Calculates BMI and gives related recommendations
- `Diet_Planner`: Suggests meals based on BMI and dietary needs
- `Workout_Scheduler`: Generates a weekly fitness plan
- `User_Proxy`: Collects and routes user data to other agents

## 🎓 Key Concepts

- **GroupChatManager**: Coordinates agent conversation
- **Autogen Agent Architecture**: Enables multi-agent collaboration
- **Gemini Flash API**: Powers LLM responses

## 🚫 Error Handling

- Displays helpful errors if:

  - API key is invalid
  - Inputs are incorrect
  - Gemini service is unreachable

## 🔗 Example Usage

```text
Weight: 70kg
Height: 170cm
Age: 30
Gender: Male
Dietary Preference: Vegan
```

Output:

- BMI = 24.2 (Normal)
- Recommendation: Maintain diet & exercise
- Meal Plan: Vegan meals with macros
- Workout: 5-day plan with cardio and strength

## 🏆 Output

- Conversation view of agent responses
- Final plan section with **download** option

---

Built using ✨ Streamlit + AutoGen + Gemini 1.5 Flash.

**Author:**   VIGNESHWARAN A
