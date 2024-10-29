# from flask import Flask, render_template, request, redirect, url_for, send_file
# import os
# from process_video import main as main_differentiated, main_simple
#
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['ALLOWED_VIDEO_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}
# app.config['ALLOWED_AUDIO_EXTENSIONS'] = {'wav', 'mp3', 'flac'}
#
#
# def allowed_file(filename, allowed_extensions):
#     """Check if file is of allowed type."""
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
#
#
# @app.route('/')
# def index():
#     """Render the homepage with the upload options."""
#     return render_template('index.html')
#
#
# @app.route('/upload_simple_video', methods=['POST'])
# def upload_simple_video():
#     """Handle video file upload and process without differentiating speakers."""
#     return handle_upload('video', main_simple, app.config['ALLOWED_VIDEO_EXTENSIONS'], "Simple")
#
#
# @app.route('/upload_differentiated_video', methods=['POST'])
# def upload_differentiated_video():
#     """Handle video file upload and process with differentiating speakers."""
#     return handle_upload('video', main_differentiated, app.config['ALLOWED_VIDEO_EXTENSIONS'], "Differentiated")
#
#
# @app.route('/upload_simple_audio', methods=['POST'])
# def upload_simple_audio():
#     """Handle audio file upload and process without differentiating speakers."""
#     return handle_upload('audio', main_simple, app.config['ALLOWED_AUDIO_EXTENSIONS'], "Simple")
#
#
# @app.route('/upload_differentiated_audio', methods=['POST'])
# def upload_differentiated_audio():
#     """Handle audio file upload and process with differentiating speakers."""
#     return handle_upload('audio', main_differentiated, app.config['ALLOWED_AUDIO_EXTENSIONS'], "Differentiated")
#
#
# def handle_upload(file_key, processing_function, allowed_extensions, version):
#     """Common function to handle file upload and processing."""
#     if file_key not in request.files:
#         return redirect(url_for('index'))
#
#     file = request.files[file_key]
#
#     if file and allowed_file(file.filename, allowed_extensions):
#         filename = file.filename
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#
#         # Process the uploaded file
#         output_file = processing_function(filepath)
#
#         return render_template('result.html', transcription_file=output_file, version=version)
#
#     return redirect(url_for('index'))
#
#
# @app.route('/download/<filename>')
# def download_file(filename):
#     """Allow users to download the transcription file."""
#     return send_file(f'static/uploads/{filename}', as_attachment=True)
#
#
# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from process_video import main as main_differentiated, main_simple, summarize_text
from process_video import summarize_text

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_VIDEO_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}
app.config['ALLOWED_AUDIO_EXTENSIONS'] = {'wav', 'mp3', 'flac'}
app.config['ALLOWED_TEXT_EXTENSIONS'] = {'txt'}


def allowed_file(filename, allowed_extensions):
    """Check if file is of allowed type."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/')
def index():
    """Render the homepage with the upload options."""
    return render_template('index.html')


@app.route('/upload_simple_video', methods=['POST'])
def upload_simple_video():
    """Handle video file upload and process without differentiating speakers."""
    return handle_upload('video', main_simple, app.config['ALLOWED_VIDEO_EXTENSIONS'], "Simple")


@app.route('/upload_differentiated_video', methods=['POST'])
def upload_differentiated_video():
    """Handle video file upload and process with differentiating speakers."""
    return handle_upload('video', main_differentiated, app.config['ALLOWED_VIDEO_EXTENSIONS'], "Differentiated")


@app.route('/upload_simple_audio', methods=['POST'])
def upload_simple_audio():
    """Handle audio file upload and process without differentiating speakers."""
    return handle_upload('audio', main_simple, app.config['ALLOWED_AUDIO_EXTENSIONS'], "Simple")


@app.route('/upload_differentiated_audio', methods=['POST'])
def upload_differentiated_audio():
    """Handle audio file upload and process with differentiating speakers."""
    return handle_upload('audio', main_differentiated, app.config['ALLOWED_AUDIO_EXTENSIONS'], "Differentiated")


def handle_upload(file_key, processing_function, allowed_extensions, version):
    """Common function to handle file upload and processing."""
    if file_key not in request.files:
        return redirect(url_for('index'))

    file = request.files[file_key]

    if file and allowed_file(file.filename, allowed_extensions):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the uploaded file
        output_file, _ = processing_function(filepath)

        # Generate a summary automatically
        with open(output_file, "r") as f:
            text_content = f.read()
        summary = summarize_text(text_content)

        # Render the result page with the automatically generated summary
        return render_template('result.html', transcription_file=output_file, version=version, summary=summary)

    return redirect(url_for('index'))


@app.route('/upload_text', methods=['POST'])
def upload_text():
    """Handle text file upload for summarization."""
    if 'textfile' not in request.files:
        return redirect(url_for('index'))

    file = request.files['textfile']

    if file and allowed_file(file.filename, app.config['ALLOWED_TEXT_EXTENSIONS']):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read the text from the file and generate the summary
        with open(filepath, "r") as f:
            text_content = f.read()

        # Use the updated summarize_text function
        summary = summarize_text(text_content)
        return render_template('result.html', transcription_file=filename, version="Uploaded Text", summary=summary)

    return redirect(url_for('index'))


@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    """Generate a summary from the existing transcription file."""
    transcription_file = request.form.get('transcription_file')

    # Debug: Check if we are receiving the correct file name
    if not transcription_file:
        print("No transcription file received.")
        return redirect(url_for('index'))

    # Read the transcription from the file and generate a summary
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], transcription_file)
    if not os.path.exists(filepath):
        print("Transcription file does not exist.")
        return redirect(url_for('index'))

    with open(filepath, "r") as f:
        text_content = f.read()

    summary = summarize_text(text_content)

    # Debug: Check if summary is successfully generated
    if not summary:
        print("Failed to generate summary.")
        return redirect(url_for('index'))

    # Render the result page with the generated summary
    return render_template('result.html', transcription_file=transcription_file, version="Transcription",
                           summary=summary)


@app.route('/download/<filename>')
def download_file(filename):
    """Allow users to download the transcription file."""
    return send_file(f'static/uploads/{filename}', as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)