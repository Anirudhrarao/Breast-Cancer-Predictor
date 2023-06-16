from flask import Flask, render_template,request, jsonify
from src.pipelines.prediction_pipeline import CustomClass,PredictionPipeline

app = Flask(__name__)

@app.route('/',methods = ['GET','POST'])
def predictor():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomClass(
                mean_radius = float(request.form.get('mean_radius')),
                mean_texture = float(request.form.get('mean_texture')),
                mean_smoothness = float(request.form.get('mean_smoothness')),
                mean_compactness = float(request.form.get('mean_compactness')),
                mean_symmetry = float(request.form.get('mean_symmetry')),
                mean_fractal_dimension = float(request.form.get('mean_fractal_dimension')),
                radius_error = float(request.form.get('radius_error')),
                texture_error = float(request.form.get('texture_error')),
                smoothness_error = float(request.form.get('smoothness_error')),
                compactness_error = float(request.form.get('compactness_error')),
                concavity_error = float(request.form.get('concavity_error')),
                concave_points_error = float(request.form.get('concave_points_error')),
                symmetry_error = float(request.form.get('symmetry_error')),
                fractal_dimension_error = float(request.form.get('fractal_dimension_error')),
                worst_smoothness = float(request.form.get('worst_smoothness')),
                worst_symmetry = float(request.form.get('worst_symmetry')),
                worst_fractal_dimension = float(request.form.get('worst_fractal_dimension'))
        )
        final_data = data.get_data_as_dataFrame()
        pipeline_prediction = PredictionPipeline()
        pred = pipeline_prediction.predict(final_data)
        result = pred[0]
        if result == 0:
            return render_template('index.html',result="Malignant: cancerous tumor")
        else:
            return render_template('index.html',result="Benign: Not a cancerous tumor")
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)