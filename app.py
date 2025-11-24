from flask import Flask

from controllers.hepatitis_controller import hepatitis_bp


def create_app() -> Flask:
    """App factory that wires controllers and views."""
    app = Flask(__name__, template_folder="Templates")
    app.register_blueprint(hepatitis_bp)
    return app


app = create_app()


if __name__ == "__main__":
    app.run()
