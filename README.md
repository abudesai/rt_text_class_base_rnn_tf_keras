Bidirectional RNN in TensorFlow/Keras for Text Classification - Base problem category as per Ready Tensor specifications.

- tensorflow
- keras
- recurrent neural network
- pandas
- numpy
- activation
- python
- adam optimizer
- text classification
- scikit-optimize
- flask
- nginx
- uvicorn
- docker
- tf-idf
- nltk
- gru
- lstm

This Bidirectional Recurrent Neural Network (RNN) is comprised of 5 layers which includes embedding, global max pooling, relu, and either GRU or LSTM layers. The model uses the Adam Optimizer to evaluate the performance of the model.

The data preprocessing step includes tokenizing the input text and converting to embedding indices. In regards to processing the labels, a label encoder is used to turn the string representation of a class into a numerical representation.

The Hyperparameter Tuning (HPT) involves finding the optimal RNN for this data (GRU vs LSTM), size of the embeddings, latent dimension of the data, and learning rate for the Adam optimizer.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as clickbait, drug, and movie reviews as well as spam texts and tweets from Twitter.

The main programming language is Python. The main algorithm is created using TensorFlow and Keras while Scikitlearn is used to calulate the model metrics and preprocess the data. NLTK, pandas, and numpy are used to help preprocess the data while Scikit-Optimize is used for HPT. Flask, Nginx, gunicorn are used to allow for web services. The web service provides two endpoints- /ping for health check and /infer for predictions in real time.
