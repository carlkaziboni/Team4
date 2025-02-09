import pandas as pd
import os

from flask import Flask, render_template, request, jsonify
from graphgen import plot
from agi import analyse

app = Flask(__name__)

current_text = "This is the initial text."
df = pd.read_csv('Diageo_Scotland_Full_Year_2024_Daily_Data.csv')

@app.route('/')
def index():
    return render_template('Gradient.html', text=current_text)

@app.route('/update_text', methods=['POST'])
def update_text():
    global current_text

    data = request.json
    img_src = data.get('largeImage')

    if img_src.startswith("http://") or img_src.startswith("https://"):
        img_filename = os.path.basename(img_src)  # Extract filename (e.g., 'graph.png')
        image_path = os.path.join("static", img_filename)  # Convert to local path
    else:
        image_path = img_src  # Already a valid local path

    new_text = analyse(image_path)
    print(new_text)
    current_text = new_text
    print(current_text)
    return jsonify({"status": "success", "new_text": current_text})

@app.route('/submit_dates', methods=['POST'])
def submit_dates():
    data = request.json
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    site = data.get("site")
    category = data.get("category")
    
    print(f"Received HIUIII Start Date: {start_date}, End Date: {end_date}")  # Debugging output
    
    plot(df, start_date, end_date, category, site)

    print(f"Received Start Date: {start_date}, End Date: {end_date}")  # Debugging output
    return jsonify({"status": "success", "message": "Dates received!", "start_date": start_date, "end_date": end_date})


if __name__ == '__main__':
    app.run(debug=True)