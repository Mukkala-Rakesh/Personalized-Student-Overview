# Personalized Student Overview
Online Judge (OJ) systems automate grading of programming assignments, making it fast and accurate. OJ systems usually provide limited feedback, only indicating if the code is correct. Using Educational Data Mining (EDM), we analyze submission data to predict student success and behaviors. Key data includes the number of submissions and submission timing. We use Multi-Instance Learning (MIL) and Explainable Artificial Intelligence (XAI) to make predictions understandable. Tested on data from three years with 2,500 submissions, our method accurately models student profiles and predicts success.
![image](https://github.com/user-attachments/assets/094b86b0-7b9c-4bb0-b2e6-45497f509112)

![image](https://github.com/user-attachments/assets/782342ed-4eca-40e8-ab60-eed75aec6f96)


Registration Process The registration process involves filling out a form with various fields to create an account on the platform. Here are the steps and details for each field:

Enter Username:

Input your desired username in the provided text box. Example: "Yogesh" Enter Email Id:

Enter a valid email address. Example: "mail@gmail.com" Enter Gender:

Select your gender from the dropdown menu. Example: "Male" Enter Country Name:

Type the name of your country. Example: "India" Enter City Name:

Type the name of your city. Example: "City" Enter Password:

Create a password for your account. Example: (hidden for security) Enter Address:

Provide your address details. Example: "address" Enter Mobile Number:

Enter your mobile number. Example: "5656468" Enter State Name:

Type the name of your state. Example: "state" Register:

After filling out all the fields, click the "REGISTER" button to submit your details and create an account.
![image](https://github.com/user-attachments/assets/f096d367-2031-47eb-80f9-f54a0f670057)

![image](https://github.com/user-attachments/assets/9851410a-6319-4c2b-9bf3-0e616f3936bb)


Inputs The form requires several inputs related to the student:

Fid (ID): A unique identifier for the student. Gender: The gender of the student. Parental Level of Education: The highest education level achieved by the student's parents. Race/Ethnicity: The race or ethnicity group the student belongs to. Math Score: The student's score in mathematics. Writing Score: The student's score in writing. Solving Tasks by Time: The number of tasks solved by the student within a given time. Age: The age of the student. Lunch: The type of lunch the student receives (standard or free/reduced). Degree Type (degree_t): The type of degree the student is pursuing (e.g., Science & Technology). Test Preparation Course: Whether the student has completed a test preparation course. Reading Score: The student's score in reading. Internships: The number of internships the student has completed. Tasks Submitted on Date: The number of tasks submitted by the student on a specific date. Prediction Once these inputs are filled in, the system uses a trained dataset to predict the student's profile. This process involves:

Data Collection: The system collects data about students, including their scores, demographics, and other relevant information. Training the Model: Using machine learning algorithms, the system is trained on this dataset to learn patterns and relationships between the inputs and the desired output (student profile). Making Predictions: When new data is entered into the form, the system uses the trained model to predict the student's profile. This could include various aspects such as academic performance, likelihood of success in certain subjects, or other profile-related predictions. Explainable Artificial Intelligence (XAI) Explainable AI is crucial here as it allows the predictions made by the system to be understood by humans. This means the system can provide insights into why a particular prediction was made, based on the input features. This transparency helps in:

Building Trust: Users can understand and trust the predictions. Identifying Bias: Ensuring that the model is fair and not biased towards any group. Improving the Model: Providing feedback to improve the accuracy and fairness of the model

![image](https://github.com/user-attachments/assets/5f3c2748-5179-485d-9ef3-b26c4ce81add)


The navigation bar (navbar) at the top of the page contains several options. Here's an explanation of each option in the navbar:

Navbar Options Browse Students Datasets and Train & Test Data Sets

Explanation: This option allows users to browse through the datasets that contain information about students. It includes both the training datasets (used to train the machine learning models) and the testing datasets (used to evaluate the model's performance). View Trained and Tested Accuracy in Bar Chart

Explanation: This option provides a visual representation of the accuracy of the trained models. The bar chart displays the accuracy metrics, helping users understand how well the models perform on the training and testing datasets. View Trained and Tested Accuracy Results

Explanation: Similar to the bar chart option, this provides detailed accuracy results in a more comprehensive format. Users can view numerical accuracy metrics and possibly other evaluation metrics (e.g., precision, recall) for the trained models. View Prediction Of Online Student's Profile Judgement

Explanation: This option allows users to see the predictions made by the system for students' profiles. It uses the input data provided to generate predictions about various aspects of the students' profiles. View Online Student's Profile Judgement Ratio

Explanation: This option shows the ratio of different judgments made by the system. It provides an overview of how many students fall into different profile categories based on the predictions. Download Predicted Data Sets

Explanation: This option enables users to download the datasets containing the predicted profiles of students. It is useful for further analysis or record-keeping. View Online Student's Profile Judgement Type Ratio Results

Explanation: This option provides detailed ratio results of different types of judgments made by the system. It may include statistical summaries or visual representations of the distribution of profile types. View All Remote Users

Explanation: This option displays a list of all remote users who have used the system. The list includes user names, email addresses, gender, addresses, mobile numbers, countries, states, and cities, as shown in the main section of the image. Logout

Explanation: This option logs the user out of the system. It is a standard feature in web applications to ensure security and privacy.

![image](https://github.com/user-attachments/assets/62672505-9304-4c29-8c9c-18b69949b673)


Model Types and Accuracy The table lists various machine learning models along with their corresponding accuracy scores. Here are the models and their accuracies as shown in the table:

Artificial Neural Network (ANN)

Accuracy: 56.00000000000001 Explanation: ANNs are computing systems inspired by the biological neural networks that constitute animal brains. They are used for pattern recognition and classification tasks. Naive Bayes

Accuracy: 63.0 Explanation: Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features. Support Vector Machine (SVM)

Accuracy: 60.5 Explanation: SVMs are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. They are effective in high-dimensional spaces. Logistic Regression

Accuracy: 64.0 Explanation: Logistic regression is a statistical model that uses a logistic function to model a binary dependent variable. It's used for binary classification problems. Gradient Boosting Classifier

Accuracy: 59.5 Explanation: Gradient Boosting is a machine learning technique for regression and classification problems, which builds a model in a stage-wise fashion from weak learners (usually decision trees). Decision Tree Classifier

Accuracy: 60.5 Explanation: Decision Trees are a non-parametric supervised learning method used for classification and regression. They predict the value of a target variable by learning simple decision rules inferred from the data features. K-Neighbors Classifier

Accuracy: 51.0 Explanation: The K-Nearest Neighbors (KNN) algorithm is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. Interpretation Accuracy Scores: The accuracy score represents the percentage of correct predictions made by the model. Higher accuracy indicates better performance. Comparison: Users can compare the performance of different models to choose the most effective one for predicting student profiles. For example, Logistic Regression has the highest accuracy at 64.0, while K-Neighbors Classifier has the lowest at 51.0. Model Selection: Depending on the context and requirements, users might prioritize different models. For instance, a model with a slightly lower accuracy might be preferred if it is faster to train and predict.

![image](https://github.com/user-attachments/assets/bb7986a2-70dd-4f1d-b9c3-83993ffe2161)


Bar Graph: The main part of the page displays a bar graph comparing the accuracy of different machine learning models in predicting student profiles: X-axis: The models are listed: Artificial Neural Network-ANN, Naive Bayes, SVM, Logistic Regression, Gradient Boosting Classifier, Decision Tree Classifier, and KNeighbors Classifier. Y-axis: Represents the accuracy of each model, ranging from 50% to 66%. Bar Heights: Each bar shows the accuracy of a particular model, indicating its performance in predicting student profiles.

![image](https://github.com/user-attachments/assets/3eb5c09b-6e88-46cc-9887-b32a9b45290e)


![image](https://github.com/user-attachments/assets/5735166a-db82-4053-b342-ed7c53cdd869)


![image](https://github.com/user-attachments/assets/9375fefe-daa4-4808-b9f9-5ef37eb50a2a)

This table shows the ratio of online student judgements for a given student profile. For example, 77.77% of the judgements for a certain student profile were "Excellent" and 22.22% of the judgements for that same student profile were "Poor".
![image](https://github.com/user-attachments/assets/957462f2-2336-4d22-8603-2cc4275379c7)


Online Judge (OJ) systems are widely used in programming courses for their fast and objective grading of student submissions. However, these systems typically only indicate whether a submission meets the assignment requirements, offering little additional feedback. This limitation could be addressed by extracting more insights from the data collected by OJ systems, such as student habits, behavior patterns, and profiles related to task success or failure.To overcome this, our work applies Educational Data Mining (EDM) techniques, specifically Multi-Instance Learning (MIL) and classical Machine Learning (ML),..We evaluated our methodology using data from a programming course in a Computer Science degree, including over 2,500 submissions from around 90 students across three academic years. Our model accurately predicts whether students will pass or fail based on their submission behavior and identifies at-risk student groups. This feedback can help both students and instructors
