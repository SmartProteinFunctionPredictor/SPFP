import urllib.parse
import ast
from model import final_pred, contains_valid_amino_acids
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form values
        value1 = request.form.get('inputFormat')
        value2 = request.form.get('value2')
        value3 = request.form.get('value3')
        print(value3)
        if value1 == 'fasta':
            value3 = value3.split()[-1].upper()
            sequ = contains_valid_amino_acids(value3)
            print(sequ)

        print(value3)
        predictions = final_pred(value3.upper(), float(value2))
        prediction_result = predictions
        return redirect(url_for('prediction', result=prediction_result))


@app.route('/prediction/<result>')
def prediction(result):
    url_data = result
    embed_link = 'https://www.ebi.ac.uk/QuickGO/term/'
    data_list = ast.literal_eval(urllib.parse.unquote(url_data))

    df = pd.DataFrame(data_list, columns=['Predicted Function', 'Predicted Value (Confidence)'])
    df_go = df['Predicted Function'].apply(lambda x: x.split('-')[0])
    pd.set_option('display.max_colwidth', None)

    df['Go Term'] = '<a href="' + embed_link + df_go + '" target="_blank">' + df_go + '</a>'

    df = df[['Go Term', 'Predicted Function', 'Predicted Value (Confidence)']]
    df['Predicted Function'] = df['Predicted Function'].apply(lambda x: x.split('-')[1]).to_frame()

    html_table_with_centered_headers = df.to_html(classes='table table-bordered table-striped', border=0, index=False,
                                                  escape=False, render_links=True)
    html_table_with_centered_headers = html_table_with_centered_headers.replace('<th>',
                                                                                '<th style="text-align: center;">')
    return render_template('Prediction.html', result=html_table_with_centered_headers)


if __name__ == '__main__':
    app()
