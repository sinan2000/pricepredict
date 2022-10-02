This is a project that I really like. I am passionate about the stock market, so I think
this was a good opportunity to combine two hobbies. Using Long short-term memory,
this program tries to predict the price movement of a stock in its' near future. This requires
some libraries, for accessing the API that provides the dataset of the stock, parsing
the json file that comes as dataset, verifying if the dataset is already saved locally
or it has to be downloaded, then to be automatically saved for future use, creating the dataframe, 
the neural network and the visualization of the price movement.
This is not a personal idea, I got inspired from more sources over the internet, but it helped
me gain knowledge of quite some things about Machine Learning, optimizing Neural Networks and
even improve in Mathematics. 
I also migrated from the tensorflow code I've originally found on the internet,
so you can see quite some warnings if you decide to download and run the program. I
shall resolve all those things in the next update.
Ideas for new update:
1. creating the possibility of looking into criptocurrency as well
2. solving all the problems that cause the final figures' unallignment
3. looking further into improving the algorithm that predicts the prices

One interesting research that I'll study and I think could be of help:
https://www.sciencedirect.com/science/article/pii/S2666827022000378

Original article where I've seen the idea: (done in Tensorflow 1.6, that code does not work anymore)
https://www.datacamp.com/tutorial/lstm-python-stock-market

API used for datasets of stocks' prices:
https://www.alphavantage.co/