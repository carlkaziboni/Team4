from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

current_text = "This is the initial text."

@app.route('/')
def index():
    return render_template('Gradient.html', text=current_text)

@app.route('/update_text', methods=['POST'])
def update_text():
    global current_text
    new_text = request.form['text']
    current_text = new_text
    return jsonify({"status": "success", "new_text": current_text})

if __name__ == '__main__':
    app.run(debug=True)