from flask import Blueprint, jsonify, render_template, request

from Models.predictor import (
    ArtifactLoadError,
    BINARY_FEATURES,
    BINARY_NO_VALUE,
    BINARY_YES_VALUE,
    HepatitisPredictor,
)

hepatitis_bp = Blueprint("hepatitis", __name__)
predictor = HepatitisPredictor()


@hepatitis_bp.route("/", methods=["GET", "POST"])
def formulario():
    error = None
    result = None
    values = predictor.example_payload()

    if request.method == "POST":
        values = {}
        for feature in predictor.feature_order:
            if feature in BINARY_FEATURES:
                values[feature] = (
                    BINARY_YES_VALUE if request.form.get(feature) else BINARY_NO_VALUE
                )
            else:
                values[feature] = request.form.get(feature, "")
        try:
            result = predictor.predict(values)
        except KeyError as missing:
            error = f"Falta el campo {missing.args[0]}"
        except ValueError:
            error = "Todos los valores deben ser numericos."
        except ArtifactLoadError:
            error = "Los modelos no estan cargados."
        except Exception as exc:  # pragma: no cover - UI guardrail
            error = f"No se pudo generar la prediccion: {exc}"

    return render_template(
        "hepatitis_form.html",
        feature_order=predictor.feature_order,
        values=values,
        result=result,
        error=error,
        model_ready=predictor.ready,
        startup_error=predictor.startup_error,
        binary_fields=BINARY_FEATURES,
        binary_yes=BINARY_YES_VALUE,
        binary_no=BINARY_NO_VALUE,
    )


@hepatitis_bp.route("/api/hepatitis/health", methods=["GET"])
def health():
    status = "ok" if predictor.ready else "error"
    return (
        jsonify(
            {
                "status": status,
                "model_loaded": predictor.ready,
                "expected_features": predictor.feature_order,
                "error": str(predictor.startup_error) if predictor.startup_error else None,
            }
        ),
        200 if predictor.ready else 500,
    )


@hepatitis_bp.route("/api/hepatitis/schema", methods=["GET"])
def schema():
    if not predictor.ready:
        return (
            jsonify({"error": "Los modelos no estan cargados", "details": str(predictor.startup_error)}),
            500,
        )

    return jsonify(predictor.schema()), 200


@hepatitis_bp.route("/api/hepatitis/predict", methods=["POST"])
def predict_api():
    if not predictor.ready:
        return jsonify({"error": "Los modelos no estan cargados"}), 500

    if not request.is_json:
        return jsonify({"error": "El cuerpo debe ser JSON"}), 400

    payload = request.get_json()

    try:
        result = predictor.predict(payload)
    except KeyError as missing:
        return jsonify({"error": f"Falta el campo {missing.args[0]}"}), 400
    except ValueError:
        return jsonify({"error": "Todos los valores deben ser numericos"}), 400
    except Exception as exc:  # pragma: no cover - runtime guardrail
        return jsonify({"error": "No se pudo procesar la solicitud", "details": str(exc)}), 500

    return jsonify(result), 200
