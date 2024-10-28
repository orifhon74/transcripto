from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from process_video import main as main_differentiated, main_simple

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}


def allowed_file(filename):
    """Check if file is of allowed type."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Render the homepage with the upload options."""
    return render_template('index.html')


@app.route('/upload_simple', methods=['POST'])
def upload_simple():
    """Handle file upload and process without differentiating speakers."""
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process video without speaker differentiation
        output_file = main_simple(filepath)

        return render_template('result.html', transcription_file=output_file, version="Simple")

    return redirect(url_for('index'))


@app.route('/upload_differentiated', methods=['POST'])
def upload_differentiated():
    """Handle file upload and process with differentiating speakers."""
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process video with speaker differentiation
        output_file = main_differentiated(filepath)

        return render_template('result.html', transcription_file=output_file, version="Differentiated")

    return redirect(url_for('index'))


@app.route('/download/<filename>')
def download_file(filename):
    """Allow users to download the transcription file."""
    return send_file(f'static/uploads/{filename}', as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)