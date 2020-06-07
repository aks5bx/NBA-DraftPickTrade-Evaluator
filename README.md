# NBA-DraftPickTrade-Evaluator

## Project Overview

Program to evaluate NBA trades involving draft picks. Essential idea is to predict the win shares of a certain draft position (this is done using a simple exponential regression equation) and then compare the total win shares of draft packages against the total win shares of draft packages being offered in a trade request. For example, if draft package A has X win shares in it and draft package B has Y win shares in it, if a team offers package A to another team in exchange for package B, a comparison of the X and Y values will indicate who is getting the better end of the trade. 

Edit: The model has been expanded to now include Player-Pick trades in addition to just Pick-Pick swaps

## Technology Used
This program is implemented in python using seaborn, scipy, numpy, and pandas. The program also contains some visualizations in order to justify the exponential regression equation chosen. In addition, the program contains some test functions to show that the functions work. 

## Current Issues/Future Improvements
1. The model seems to overrate players and underrate draft picks - smoothing outliers could help with this. 
2. The model is based on WS, but does not account for "quality" WS (for example a WS gained via the playoffs vs a low-stakes regular season game) 
