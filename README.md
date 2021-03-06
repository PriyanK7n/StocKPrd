

 <img src="icon.png" align="right" />

# Google Stock Price Prediction  problem [![Stock](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome#readme)
> My Plan Of Action


##I am   interested in the directions taken by our predictions, rather than the closeness of their values to the real stock price. 
 
 - [Check My Predictions] - I  follow the same directions as the real stock price and not  really care whether my  predictions are close the real stock price. 
 The predictions could indeed be close but often taking the opposite direction from the real stock price.
 


##Steps to  improve this RNN model


-[Getting more training data]- We trained our model on the past 5 years of the Google Stock Price but it would be even better to train it on the past 10 years.

-[Increasing the number of timesteps]- the model remembered the stock prices from the 60 previous financial days to predict the stock price of the next day. That’s because we chose a number of 60 timesteps (3 months). You could try to increase the number of timesteps, by choosing for example 120 timesteps (6 months).

-[Adding some other indicators]-if you have the financial instinct that the stock price of some other companies might be correlated to the one of Google, you could add this other stock price as a new indicator in the training data.

-[Adding more LSTM layers] - we built a RNN with four LSTM layers but you could try with even more.

-[Adding more neurones in the LSTM layers] - we highlighted the fact that we needed a high number of neurones in the LSTM layers to respond better to the complexity of the problem and we chose to include 50 neurones in each of our 4 LSTM layers. 
 You could try an architecture with even more neurones in each of the 4 (or more) LSTM layers.



##Parameter Tuning on the RNN model
> Improvements in Loss


