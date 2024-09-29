# CNN-LSTM_FOR_MALARIA_DETECTION
 # PERFORMANCE ANALYSIS OF CONVOLUTIONAL NEURAL NETWORK AND LONG-SHORT MEMORY NETWORKS ON MALARIA DISEASE

Malaria, an infectious disease that has claimed the lives of many worldwide, is transmitted by the female mosquito Anopheles. Early detection of malaria parasites in the cell lowers the risk of death and improves the patient's chances of receiving prompt treatment. The study aimed to compare the accuracies of deep neural networks. This study used deep learning approaches such as CNN (convolutional neural networks) and LSTM (Long Short Term Memory networks) to detect malaria parasites, using performance metrics. A total of 27,578 images were used for this research. The images were resized, classified, preprocessed, and normalized.

Python packages and libraries like keras, numpy, matplotlib, seaborn, sklearn and tensorflow were used in the implementation phase. Google colab was the Integrated Development Environment used in this work. 
The CNN had an accuracy of 95% while the LSTM had an accuracy of 68%. Results are not determined based on just accuracy alone. The CNN had a precision of 95%, F1 score of 95.5%,  and a sensitivity of 96% while the LSTM had a precision of 78%, F1 score of 64%, sensitivity of 55%.  After models have been trained and predictions made, the CNN performed better than the LSTM in all performance metrics. 

In conclusion, this study aimed to analyze the performance of two deep learning algorithms, Convolutional Neural Network (CNN) and Long Short Term Memory Network (LSTM), for the diagnosis of malaria through image analysis. The results demonstrated that the CNN model outperformed the LSTM model, achieving a higher accuracy of 95% and 95% precision compared to 68% accuracy and 78% precision for the LSTM model. The limited dataset used in this analysis consisted of parasitic and non-parasitic blood cell images obtained from lab scans of a kaggle.com dataset. CNN is proven to achieve the most accurate results based on all performance metrics ranging from accuracy to precision to sensitivity/recall and F1 score.These findings suggest that CNN is more effective in the diagnosis of malaria using image analysis.


# Prerequisites
Keras 2.2.0

Tensorflow-GPU 1.9.0

Scikit-Learn

OpenCV

Matplotlib


