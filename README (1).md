![Voice](https://media3.giphy.com/media/3ohhwJlP6YTxY8F4is/giphy.gif)

# _**Identifying Male or Female based on voice using XGBoost and ML**_
natural language has taken over the world with voice assistants like Alexa, Siri, Cortana etc, all serve the same purpose that is speech recognition and assisting the humans in all kinds of ways. Although the front end of this might look pretty and easier to get answers but the backend which involving in how does the data given by humans pre-processed and given back to them Is very complex to understand. Similarly, Identifying thegender of a person using audio files might get tricky but it is possible if we have a proper dataset. The input which we give to the model will be in the type of csv format of wav file. The input data consists of frequency of audio, modulation index and some other important factors responsible for voice audio of a person. More than 3 models are trained to compare which is performing better and giving us the best results. (KNearest Neighbour, Artificial Neural Network and Boost)

# _**Base Paper**_
+ https://www.researchgate.net/publication/321479309_Voice_based_gender_classification_using_machine_learning
+ https://www.researchgate.net/publication/312219824_Voice_Gender_Recognition_Using_Deep_Learning

# _**Algorithm Description**_

# **Extreme Gradient Boosting Classifier:**
Extreme Gradient Boosting is a boosting algorithm which is introduced as a library written in C++, to optimize the existing gradient boosting algorithm. In this algorithm, decision trees are sued sequentially and weights are assigned to each independent variable. The weights are increased for the variables of the tree which has predicted wrong class and these variables are fed to the second tree. Further down, all these individuals’ classifiers/predictors then ensemble to give a more precise result. This algorithm can work with regression and classification.

![GBS](https://ars.els-cdn.com/content/image/1-s2.0-S2090447921000125-gr4.jpg)

# **Nearest Neighbour:**
KNN or K Nearest neighbours is a basic yet an efficient algorithm which is being used in most of the Machine learning application. Since it is a non-parametric i.e. This algorithm doesn’t make any underlying assumption like other algorithms do, such as having specify distribution of data to work with. So, this makes it very easy and understandable to all the users who are using it. The Technique KNN applies in predicting on new data is where it finds the nearest neighbours for the given point and takes a majority voting, whichever class is resided near to the new point, it will be considered as the new class for the new data point.

![KNN](https://intuitivetutorial.com/wp-content/uploads/2023/04/knn-1.png)

# **Artificial Neural Network:**
ANN is a neural network which tries to perform tasks like a human does, think like a human brain. Just like a human brain understands things after learning by watching things or by experience, ANN does the same as well. It learns with experience of going through the dataset multiple times and understands the relations, hidden features and parameters. ANN is helpful in doing regression, classification tasks and performs extremely well on huge datasets achieving high accuracy.

**Input Layer:**
Whatever input you pass for the model to learn goes through this layer of neural network for performing calculations. 

**Hidden Layer:**
The layer as the name suggests hidden because when we see the real time application we only focus on the input and output, we do not focus on how things happen. Hidden layer performs calculations, does processing, understands the hidden features and updates weights to get the best possible accuracy.

**Output Layer:**
The input passes through hidden layer where processing happens and output is returned.

![ANN](https://editor.analyticsvidhya.com/uploads/210362021-07-18%20(2).png)

# _**How to Execute?**_
So, before execution we have some pre-requisites that we need to download or install i.e., anaconda environment, python and a code editor.
**Anaconda**: Anaconda is like a package of libraries and offers a great deal of information which allows a data engineer to create multiple environments and install required libraries easy and neat.

**Download link:**

![Anaconda](https://1.bp.blogspot.com/-UJ1Ws2zZ9V4/TtMbG2ynJiI/AAAAAAAABbM/m6t2kuEhKdY/s1600/The-biggest-anaconda-snake-3.jpg)

https://www.anaconda.com/

**Python**: Python is a most popular interpreter programming language, which is used in almost every field. Its syntax is very similar to English language and even children and learning it nowadays, due to its readability and easy syntax and large community of users to help you whenever you face any issues.

**Download link:**

![Python](https://i0.wp.com/reptileworldfacts.com/wp-content/uploads/2019/05/male-blonde-super-tiger-reticulated-python.jpg?resize=351%2C351&ssl=1)

https://www.python.org/downloads/

**Code editor**: Code editor is like a notepad for a programming language which allows user to write, run and execute program which we have written. Along with these some code editors also allows us to debug, which usually allows users to execute the code line by line and allows them to see where and how to solve the errors. But I personally feel visual code is very good to work with any programming language and makes a great deal of attachment with user.

**Download links:**

![Vs code](https://schwabencode.com/contents/logos/VS2019-Badge.png) ![Pycharm](https://i0.wp.com/scracked.com/wp-content/uploads/2020/01/PyCharm-2019.3.4-Crack.png?fit=200%2C200&ssl=1)

+ https://code.visualstudio.com/Download, 
+ https://www.jetbrains.com/pycharm/download/#section=windows

# _**How to create a new environment and configure jupyter notebook with it.**_
Let us define an environment and why we need different environments. An environment is a collection of libraries that are required to run our project. When we already have an environment with the necessary libraries, why do we need a new environment?
To avoid version mismatches, we create a new environment for each project. For example, in your previous project, you used "tf env" with tensorflow 2.4 and keras 2.4, but in your current project, you must use tensorflow 2.6 and keras 2.6. If you continue your project in the "tf env" environment, there will be a version mismatch and you will need to update tensorflow and keras, but this will cause problems with the previous project's execution. To avoid this, we create a new environment with tensorflow 2.6 and keras 2.6 and resume our project.

# _**How to create an environment in anaconda.**_
+ Type “conda create –n <<name_of_your_env>>”
example: conda create -n env
+ It will ask to proceed with the environment location, type ‘y’ and press enter.
+ When you press ‘y’, the environment will be created. To activate your environment type conda activate <<your_env_name>> . E.g., conda activate myenv.
+ You can see that the environment got changed after conda activate myenv line. It changed from “base” to “myenv” which means you are now working in “myenv” environment.
+ To install a library in your virtual environment type pip install <library_name>.
e.g., pip install pandas
+ Instead of installing libraries one by one you can even install by bunch, i.e., we have a txt file called requirements.tx which consists of all the libraries required to proceed with the project, so we can use it.
+ so, before installing requirements.txt, make sure you are in the specific path where your requirements.txt is located, basically this file is located in the folder where our executable files are located, so we need to move to that directory by following command.
**cd C:\folder_name**
+ Here A -> drive, folder name -> path where your executable file is saved
+ I go to that file path in anaconda using cd command 
1.	Go to drive where your project file is.
2.	Go to the path of your project using cd <path>
3.	Type pip install –r requirements.txt 
+ And all your required libraries will be downloaded and you can start your project.
+ But if you want to use jupyter notebook on the new environment you have to set it up for the new environment.
+ After you have installed all the libraries and created an environment, you need an editor to run the code, that is starting jupyter notebook, as soon as you enter jupyter notebook in the terminal you will definitely get this error. “Jupiter” is not recognized as an internal or external command.
So, to solve it it we have 2 commands.
1.	conda install –c conda-forge jupyterlab
2.	conda install –c anaconda python
Now you are ready to use jupyter on this environment and start with your project!

![thanks](https://media1.giphy.com/media/ZfK4cXKJTTay1Ava29/giphy.gif)
  
  
# _**Steps to execute**_
**Note:** Make sure you have added path while installing the software’s.(Supports Python==3.8 only)

1.	Install the prerequisites/software’s required to execute the code.
2.	Press windows key and type in anaconda prompt a terminal opens up.
3.	Before executing the code, we need to create a specific environment which allows us to install the required libraries necessary for our project.
•	Type conda create -name “env_name”, e.g.: conda create -name project_1
•	Type conda activate “env_name, e.g.: conda activate project_1
4.	Make sure you are in the correct path in your terminal, where you have saved your executable file/folder. E.g.: cd A:\project\AI\Completed\project_name, then press enter.
5.	Install necessary libraries from requirements.txt file provided.
6.	Run pip install -r requirements.txt or conda install requirements.txt (Requirements.txt is a text file consisting of all the necessary libraries required for executing this python file. If it gives any error while installing libraries, you might need to install them individually.)
7.	If you want to build your own model for detection, you can go through “lstm.ipynb”. It takes time to build the model from scratch again.
8.      Type main.ipynb to get the results.

# _**Data Description**_
The Dataset is collected form Kaggle Repository which contains 3168 Instances with 21 features. Some of the features which correlates better with the model.

![Dataset](https://miro.medium.com/v2/resize:fit:1200/1*c8wR6BtKoF-kG5lqqcI7hA.gif)

 # _**Issues Faced.**_
1. We might face an issue while installing specific libraries.
2. Make sure you have the latest version of python or 3.8, since sometimes it might cause version mismatch.
3. Adding path to environment variables in order to run python files and anaconda environment in code editor, specifically in visual studio code.

# _**Note:**_
**All the required data hasn't been provided over here. Please feel free to contact me for any issues. You can also download the dataset from the given link below.**

### _**Let’s Connect**_
<a href="https://linkedin.com/in/mudassiruddin21" target="blank"><img align="center" src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/linkedin.svg" alt="mudassiruddin21" height="30" width="40" /></a>

![Connect](https://media1.giphy.com/media/khr2lS27v92PQPD3oa/giphy.gif)

# _**Yes, you now have more knowledge than yesterday, Keep Going.**_
![Happy](https://media2.giphy.com/media/BPJmthQ3YRwD6QqcVD/giphy.gif)
  
