# app.py (tiny entrypoint)

import os
from dotenv import load_dotenv
from flask import Flask

from config import Config
from app_ext import init_app

load_dotenv()

app = Flask(__name__)
app.config.from_object(Config)
init_app(app)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)