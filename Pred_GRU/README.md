# Stock-Price-Prediction-Based-on-ML
Instruction：First execute data process.py to generate data file, then execute 5min or daily to predict stock price.                   


COMMITMENTS:            
1.Use GRU model to predict stock price         
2.Plot the price line of real and predicted price         
3.Calculate the RMES and MES as index to show the the accuracy of the model                       

RESULT:
Daily:----- Test_RMSE ----- 483.2620460907412                   
5min:----- Test_RMSE ----- 75.74034466739836         

PRESENT PROBLEMS for further work：             
1.The The functions of investment portfolios haven't been added       
2.The process of plot 5min_data is extremely slow due to the amount of data           
3.Present data has been split to train and test set, which means that validation set is same as test set, I don't know if it's ok... 

1.20update:
1.The process of plot 5min_data is extremely slow due to the amount of data(done: use list to store the predicted price makes it faster.)           
2.Present data has been split to train and test set, which means that validation set is same as test set, don't know whether it's ok...(done:split into train, val and test set.)
