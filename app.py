
# from flask import Flask, render_template, request, jsonify

# import pandas as pd
# import numpy as np
# from scipy.optimize import linprog
# from openai import OpenAI
# import os
# import json
# import logging
# from dotenv import load_dotenv
# import traceback

# # === Setup Logging ===
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler("meal_planner.log"),
#         logging.StreamHandler()
#     ]
# )

# # === Load Env & OpenAI Client ===
# load_dotenv()
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# # === Load Data ===
# try:
#     df = pd.read_csv("Tagged_Indian_Food_Nutrition.csv")
#     logging.info("CSV loaded successfully.")
# except Exception as e:
#     logging.error(" Failed to load CSV: %s", str(e))
#     raise

# df.columns = [col.strip().replace(" ", "_").replace("(", "").replace(")", "").lower() for col in df.columns]
# df.rename(columns={"food_name": "food_name_llm", "type": "veg_nonveg", "recommended_meal": "meal_time"}, inplace=True)
# df = df.dropna(subset=["protein_g_per_100kcal", "carbohydrates_g_per_100kcal", "fats_g_per_100kcal"]).reset_index(drop=True)

# # Normalize per kcal
# df["protein_per_kcal"] = df["protein_g_per_100kcal"] / 100
# df["carbs_per_kcal"] = df["carbohydrates_g_per_100kcal"] / 100
# df["fat_per_kcal"] = df["fats_g_per_100kcal"] / 100
# df["sugar_per_kcal"] = df["free_sugar_g_per_100kcal"].fillna(0) / 100
# df["fiber_per_kcal"] = df["fibre_g_per_100kcal"].fillna(0) / 100
# df["sodium_per_kcal"] = df["sodium_mg_per_100kcal"].fillna(0) / 100

# # Filter unrealistic entries
# df = df[
#     (df["protein_g_per_100kcal"] <= 25) &
#     (df["fats_g_per_100kcal"] <= 20) &
#     (df["carbohydrates_g_per_100kcal"] <= 40)
# ].copy()

# # Portion bounds
# restrict_keywords = ["sauce", "chutney", "pickle", "jam", "ketchup", "achar", "achaar", "murabba", "marmalade", "preserve", "squash", "puree", "relish", "dip"]
# df["portion_bounds"] = df["dish_name"].apply(lambda name: (0, 1) if any(k in name.lower() for k in restrict_keywords) else (0, 4))

# # === Flask App ===
# app = Flask(__name__)

# def extract_nutrition_preferences(natural_input: str):
#     try:
#         system_msg = """
# You are a dietitian AI that extracts structured nutrition goals from natural language.
# Return a JSON object with:
# - calories_target
# - protein_pct, carbs_pct, fat_pct (sum to 100)
# - fiber_min (default 25)
# - sugar_max (default 25)
# - sodium_max (default 2300)
#         """
#         user_msg = f"Input: {natural_input}"

#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": system_msg},
#                 {"role": "user", "content": user_msg}
#             ],
#             temperature=0
#         )

#         raw_output = response.choices[0].message.content.strip()
#         logging.info("LLM Raw Output:\n%s", raw_output)

#         # ✅ Remove markdown-style code block if present
#         if raw_output.startswith("```json"):
#             raw_output = raw_output.replace("```json", "").replace("```", "").strip()
#         elif raw_output.startswith("```"):
#             raw_output = raw_output.replace("```", "").strip()

#         return json.loads(raw_output)

#     except Exception as e:
#         logging.error("Error extracting preferences: %s", str(e))
#         logging.debug(traceback.format_exc())
#         raise

# @app.route('/')
# def home():
#     return render_template('home.html') 

# @app.route("/chatbot")
# def chatbot():
#     return "<h1>Chatbot coming soon!</h1>"


# @app.route("/meal-planner")
# def index():
#     return render_template("index.html")

# @app.route("/generate_meal_plan", methods=["POST"])
# def generate_meal_plan():
#     try:
#         user_input = request.json.get("input")
#         logging.info("Received Input: %s", user_input)

#         if not user_input:
#             raise ValueError("Input cannot be empty")

#         params = extract_nutrition_preferences(user_input)
#         logging.info(" Extracted Params: %s", params)

#         # Macronutrient targets
#         calories_target = params["calories_target"]
#         protein_pct = params["protein_pct"]
#         carbs_pct = params["carbs_pct"]
#         fat_pct = params["fat_pct"]
#         fiber_min = params.get("fiber_min", 25)
#         sugar_max = params.get("sugar_max", 25)
#         sodium_max = params.get("sodium_max", 2300)

#         # Convert to grams
#         protein_target = (calories_target * (protein_pct / 100)) / 4
#         carbs_target = (calories_target * (carbs_pct / 100)) / 4
#         fat_target = (calories_target * (fat_pct / 100)) / 9

#         # Extract constraints
#         protein = df["protein_g_per_100kcal"].values
#         carbs = df["carbohydrates_g_per_100kcal"].values
#         fat = df["fats_g_per_100kcal"].values
#         fiber = df["fibre_g_per_100kcal"].values
#         sugar = df["free_sugar_g_per_100kcal"].fillna(0).values
#         sodium = df["sodium_mg_per_100kcal"].fillna(0).values

#         c = np.ones(len(df)) + 0.01 * sugar + 0.001 * sodium
#         A = [
#             -protein, -carbs, -fat,
#             -fiber, sugar, sodium
#         ]
#         b = [
#             -protein_target, -carbs_target, -fat_target,
#             -fiber_min, sugar_max, sodium_max
#         ]
#         bounds = list(df["portion_bounds"])

#         result = linprog(c=c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

#         if not result.success:
#             logging.warning(" Optimization failed: %s", result.message)
#             return jsonify({"error": result.message}), 400

#         df["Portions_100kcal"] = result.x.round(2)
#         selected = df[df["Portions_100kcal"] > 0].copy()

#         selected["Calories"] = selected["Portions_100kcal"] * 100
#         selected["Total Protein (g)"] = selected["Portions_100kcal"] * selected["protein_g_per_100kcal"]
#         selected["Total Carbs (g)"] = selected["Portions_100kcal"] * selected["carbohydrates_g_per_100kcal"]
#         selected["Total Fat (g)"] = selected["Portions_100kcal"] * selected["fats_g_per_100kcal"]
#         selected["Total Fiber (g)"] = selected["Portions_100kcal"] * selected["fibre_g_per_100kcal"]
#         selected["Total Sugar (g)"] = selected["Portions_100kcal"] * selected["free_sugar_g_per_100kcal"]
#         selected["Total Sodium (mg)"] = selected["Portions_100kcal"] * selected["sodium_mg_per_100kcal"]

#         # output = selected[[
#         #     "dish_name", "veg_nonveg", "meal_time", "Portions_100kcal", "Calories",
#         #     "Total Protein (g)", "Total Carbs (g)", "Total Fat (g)",
#         #     "Total Fiber (g)", "Total Sugar (g)", "Total Sodium (mg)"
#         # ]].to_dict(orient="records")

#         output = selected[["dish_name", "veg_nonveg", "meal_time"]].to_dict(orient="records")


#         logging.info(" Meal plan generated with %d dishes", len(output))
#         return jsonify({
#             "plan": output,
#             "summary": {
#                 "calories": round(selected["Calories"].sum(), 1),
#                 "protein": round(selected["Total Protein (g)"].sum(), 1),
#                 "carbs": round(selected["Total Carbs (g)"].sum(), 1),
#                 "fat": round(selected["Total Fat (g)"].sum(), 1),
#                 "fiber": round(selected["Total Fiber (g)"].sum(), 1),
#                 "sugar": round(selected["Total Sugar (g)"].sum(), 1),
#                 "sodium": round(selected["Total Sodium (mg)"].sum(), 1)
#             }
#         })

#     except Exception as e:
#         logging.error(" Exception: %s", str(e))
#         logging.debug(traceback.format_exc())
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from scipy.optimize import linprog
from openai import OpenAI
import os
import json
import logging
from dotenv import load_dotenv
import traceback

# === Setup Logging ===
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("meal_planner.log"),
        logging.StreamHandler()
    ]
)

# === Load Env & OpenAI Client ===
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# === Load Data ===
try:
    df = pd.read_csv("Tagged_Indian_Food_Nutrition.csv")
    logging.info("CSV loaded successfully.")
except Exception as e:
    logging.error(" Failed to load CSV: %s", str(e))
    raise

df.columns = [col.strip().replace(" ", "_").replace("(", "").replace(")", "").lower() for col in df.columns]
df.rename(columns={"food_name": "food_name_llm", "type": "veg_nonveg", "recommended_meal": "meal_time"}, inplace=True)
df = df.dropna(subset=["protein_g_per_100kcal", "carbohydrates_g_per_100kcal", "fats_g_per_100kcal"]).reset_index(drop=True)

# Normalize per kcal
df["protein_per_kcal"] = df["protein_g_per_100kcal"] / 100
df["carbs_per_kcal"] = df["carbohydrates_g_per_100kcal"] / 100
df["fat_per_kcal"] = df["fats_g_per_100kcal"] / 100
df["sugar_per_kcal"] = df["free_sugar_g_per_100kcal"].fillna(0) / 100
df["fiber_per_kcal"] = df["fibre_g_per_100kcal"].fillna(0) / 100
df["sodium_per_kcal"] = df["sodium_mg_per_100kcal"].fillna(0) / 100

# Filter unrealistic entries
df = df[
    (df["protein_g_per_100kcal"] <= 25) &
    (df["fats_g_per_100kcal"] <= 20) &
    (df["carbohydrates_g_per_100kcal"] <= 40)
].copy()

# Portion bounds
restrict_keywords = ["sauce", "chutney", "pickle", "jam", "ketchup", "achar", "achaar", "murabba", "marmalade", "preserve", "squash", "puree", "relish", "dip"]
df["portion_bounds"] = df["dish_name"].apply(lambda name: (0, 1) if any(k in name.lower() for k in restrict_keywords) else (0, 4))

# === Flask App ===
app = Flask(__name__)

def extract_nutrition_preferences(natural_input: str):
    try:
        system_msg = """
You are a dietitian AI that extracts structured nutrition goals from natural language.
Return a JSON object with:
- calories_target
- protein_pct, carbs_pct, fat_pct (sum to 100)
- fiber_min (default 25)
- sugar_max (default 25)
- sodium_max (default 2300)
        """
        user_msg = f"Input: {natural_input}"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0
        )

        raw_output = response.choices[0].message.content.strip()
        logging.info("LLM Raw Output:\n%s", raw_output)

        # ✅ Remove markdown-style code block if present
        if raw_output.startswith("```json"):
            raw_output = raw_output.replace("```json", "").replace("```", "").strip()
        elif raw_output.startswith("```"):
            raw_output = raw_output.replace("```", "").strip()

        return json.loads(raw_output)

    except Exception as e:
        logging.error("Error extracting preferences: %s", str(e))
        logging.debug(traceback.format_exc())
        raise

def generate_meal_plan_natural_text(summary: dict, meal_plan: list) -> str:
    try:
        nutrition_summary = f"Calories: {summary.get('calories')} kcal\nProtein: {summary.get('protein')} g\nCarbs: {summary.get('carbs')} g\nFat: {summary.get('fat')} g\nFiber: {summary.get('fiber')} g\nSugar: {summary.get('sugar')} g\nSodium: {summary.get('sodium')} mg"
        meal_text = "\n".join([f"{m['dish_name']} ({m['veg_nonveg']}) for {m['meal_time']}" for m in meal_plan])

        prompt = f"""
You are a friendly and health-aware assistant.
Convert this nutritional summary and food list into a clear, natural language paragraph that is easy to understand.

Nutrition Summary:
{nutrition_summary}

Recommended Meals:
{meal_text}
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error("Error in natural text generation: %s", e)
        return "There was a problem generating your meal summary."

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/chatbot")
def chatbot():
    return "<h1>Chatbot coming soon!</h1>"

@app.route("/meal-planner")
def index():
    return render_template("index.html")
@app.route('/calorie-calculator', methods=['GET', 'POST'])
def calorie_calculator():
    result = None

    if request.method == 'POST':
        age = int(request.form['age'])
        gender = request.form['gender']
        height = float(request.form['height'])  # in cm
        weight = float(request.form['weight'])  # in kg
        activity = request.form['activity']

        # BMR calculation using Mifflin-St Jeor Equation
        if gender == 'male':
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161

        # Activity multipliers
        activity_map = {
            'sedentary': 1.2,
            'light': 1.375,
            'moderate': 1.55,
            'active': 1.725,
            'very_active': 1.9
        }

        tdee = bmr * activity_map.get(activity, 1.2)

        result = {
            "bmr": round(bmr, 2),
            "tdee": round(tdee, 2)
        }

    return render_template("calorie_calculator.html", result=result)
@app.route("/generate_meal_plan", methods=["POST"])
def generate_meal_plan():
    try:
        user_input = request.json.get("input")
        logging.info("Received Input: %s", user_input)

        if not user_input:
            raise ValueError("Input cannot be empty")

        params = extract_nutrition_preferences(user_input)
        logging.info("Extracted Params: %s", params)

        calories_target = params["calories_target"]
        protein_pct = params["protein_pct"]
        carbs_pct = params["carbs_pct"]
        fat_pct = params["fat_pct"]
        fiber_min = params.get("fiber_min", 25)
        sugar_max = params.get("sugar_max", 25)
        sodium_max = params.get("sodium_max", 2300)

        protein = df["protein_g_per_100kcal"].values
        carbs = df["carbohydrates_g_per_100kcal"].values
        fat = df["fats_g_per_100kcal"].values
        fiber = df["fibre_g_per_100kcal"].fillna(0).values
        sugar = df["free_sugar_g_per_100kcal"].fillna(0).values
        sodium = df["sodium_mg_per_100kcal"].fillna(0).values

        c = np.ones(len(df)) + 0.01 * sugar + 0.001 * sodium
        A = [
            -protein, -carbs, -fat,
            -fiber, sugar, sodium
        ]
        b = [
            -((calories_target * (protein_pct / 100)) / 4),
            -((calories_target * (carbs_pct / 100)) / 4),
            -((calories_target * (fat_pct / 100)) / 9),
            -fiber_min,
            sugar_max,
            sodium_max
        ]
        bounds = list(df["portion_bounds"])

        result = linprog(c=c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

        if not result.success:
            logging.warning("Optimization failed: %s", result.message)
            return jsonify({"error": result.message}), 400

        df["Portions_100kcal"] = result.x.round(2)
        selected = df[df["Portions_100kcal"] > 0].copy()

        # Replace all NaNs with defaults
        selected.fillna({
            "fibre_g_per_100kcal": 0,
            "free_sugar_g_per_100kcal": 0,
            "sodium_mg_per_100kcal": 0,
            "veg_nonveg": "Unknown",
            "meal_time": "Anytime"
        }, inplace=True)

        selected["Calories"] = selected["Portions_100kcal"] * 100
        selected["Total Protein (g)"] = selected["Portions_100kcal"] * selected["protein_g_per_100kcal"]
        selected["Total Carbs (g)"] = selected["Portions_100kcal"] * selected["carbohydrates_g_per_100kcal"]
        selected["Total Fat (g)"] = selected["Portions_100kcal"] * selected["fats_g_per_100kcal"]
        selected["Total Fiber (g)"] = selected["Portions_100kcal"] * selected["fibre_g_per_100kcal"]
        selected["Total Sugar (g)"] = selected["Portions_100kcal"] * selected["free_sugar_g_per_100kcal"]
        selected["Total Sodium (mg)"] = selected["Portions_100kcal"] * selected["sodium_mg_per_100kcal"]

        # Remove any remaining NaNs just in case
        selected = selected.replace({np.nan: None})

        output = selected[["dish_name", "veg_nonveg", "meal_time"]].to_dict(orient="records")

        def safe(x): return round(x, 1) if pd.notnull(x) and not np.isnan(x) else 0.0

        summary = {
            "calories": safe(selected["Calories"].sum()),
            "protein": safe(selected["Total Protein (g)"].sum()),
            "carbs": safe(selected["Total Carbs (g)"].sum()),
            "fat": safe(selected["Total Fat (g)"].sum()),
            "fiber": safe(selected["Total Fiber (g)"].sum()),
            "sugar": safe(selected["Total Sugar (g)"].sum()),
            "sodium": safe(selected["Total Sodium (mg)"].sum())
        }

        natural_text = generate_meal_plan_natural_text(summary, output)

        logging.info("Meal plan generated with %d dishes", len(output))
        return jsonify({
            "plan": output,
            "summary": summary,
            "natural_text": natural_text
        })

    except Exception as e:
        logging.error("Exception: %s", str(e))
        logging.debug(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
