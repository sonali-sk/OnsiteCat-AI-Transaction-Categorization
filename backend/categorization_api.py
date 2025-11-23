from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS  # Import CORS
from feature_extractor.feature_extractor import predict_spark_df
from categorization_model.categorization_model import categorize_pandas_df
from utils import normalize_description, create_explanation_column


app = Flask(__name__)
CORS(app)



# Import or define your model processing function here
# For example:
def run_model_on_dataframe(df):
    # Your model logic goes here.
    # For demo, let's just add a dummy 'category' column based on some logic.
    df['category'] = df['description'].apply(lambda x: 'Groceries' if 'Walmart' in x else 'Other')
    # You can add merchant, context_feature, explanation columns similarly
    df['merchant'] = df['description'].apply(lambda x: 'Walmart' if 'Walmart' in x else 'Unknown')
    df['context_feature'] = df['description'].apply(lambda x: 'weekly essentials' if 'Walmart' in x else 'miscellaneous')
    return df

@app.route('/categorize', methods=['POST'])
def categorize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read CSV into pandas DataFrame
        df = pd.read_csv(file)

        #processed_df = run_model_on_dataframe(df)
        normalized_desc = normalize_description(df)
        feature_extracted_df = predict_spark_df(normalized_desc)
        processed_df = categorize_pandas_df(feature_extracted_df)
        explain_df = create_explanation_column(processed_df)

        # Convert processed DataFrame to JSON records
        final_df = explain_df[["id", "description", "amount", "category", "explanation", "extracted_merchant_name", "context_feature"]]
        result = final_df.to_dict(orient='records')

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
