''' Executing this function initiates the application of emotion
    detection to be executed over the Flask channel and deployed on
    localhost:5000.
'''
# Importing flask and app to be deployed
from flask import Flask, render_template, request
from emotion_detection import emotion_detector

# Initiate the app
app = Flask("Emotion Detector")


@app.route("/emotionDetector")
def emo_detector():
    ''' This code receives the text from the HTML interface and 
        runs the emotion detector on it using the emotion_detector()
        function. The output returned shows the different emotions their corresponding scores
        and the dominant emotion of the statement. 
    '''
    # Retrieve the text to analyze from the request arguments
    text_to_analyze = request.args.get('textToAnalyze')

    # Pass the text to the emotion_detector function and store the response
    response = emotion_detector(text_to_analyze)

    #Error handling
    value = response.get('dominant_emotion')
    if value is not None:
        # Extracting outputs from the response
        anger_score = response['anger']
        disgust_score = response['disgust']
        fear_score = response['fear']
        joy_score = response['joy']
        sadness_score = response['sadness']
        dom_emo = response['dominant_emotion']
        return (f'For the given statement, the system response is '
            f'anger: {anger_score}, disgust: {disgust_score}, '
            f'fear: {fear_score}, joy: {joy_score} and '
            f'sadness: {sadness_score}. The dominant emotion is <b> {dom_emo}</b>.')

    return "Invalid text! Please try again!"


@app.route("/")
def render_index_page():
    ''' This function initiates the rendering of the main application
    page over the Flask channel
    '''
    return render_template('index.html')



''' This functions executes the flask app and deploys it on localhost:5000
    '''
if __name__ == "__main__":
    app.run(host = "0.0.0.0", port=5000)

