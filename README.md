# NBA-DraftPickTrade-Evaluator

## Project Overview

This is a program to evaluate NBA trades involving draft picks. The essential idea here is to predict the win shares of a certain draft position (this is done using a simple exponential regression equation) and then compare the total win shares of the two draft packages being offered in a hypothetical trade request. For example, let's say draft package A has X win shares in it and draft package B has Y win shares in it. If a team offers package A to another team in exchange for package B, a comparison of the X and Y values will indicate who is getting the better end of the trade. 

The model is then expanded to include Player-Pick trades in addition to just Pick-Pick swaps. This expansion compares the predicted win shares of a draft pick to the predicted win shares of a current NBA player (which is based on the player's recent performance). 

The model also is able to suggest ways to make unbalanced trades more balanced by suggesting additional compensation to the team being undercompensated. 

## Technology Used
This program is implemented in python using seaborn, scipy, numpy, and pandas. The program also contains some visualizations in order to justify the exponential regression equation chosen. In addition, the program contains some test functions to show that the functions work. 

## Findings 
At a proof of concept level, the model is able to evaluate a hypothetical trade and determine the balance of the trade proposal. In addition, the model is able to suggest measures to even the trade if it is uneven. However, this model exists only at a proof of concept level and would have to be expanded upon greatly in order to be used to make informed decisions in the NBA. 

## Current Issues/Future Improvements
1. The model seems to overrate players and underrate draft picks - smoothing outliers could help with this. 
2. The model is based on Win Shares (WS), but does not account for "quality" WS (for example a WS gained via the playoffs vs a low-stakes regular season game are valued the same) 
3. Win Shares itself is an incomplete metric to base player performance on. A better metric would also include factoring in contract value. 
