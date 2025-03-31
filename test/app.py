from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def print_color(color):
    return print(color)

@app.route('/')
def index():
    return render_template('index.html')

# New endpoint to receive the color from the frontend
@app.route('/print_color', methods=['POST'])
def receive_color():
    data = request.get_json()
    color = data.get('color', None)
    print_color(color)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
