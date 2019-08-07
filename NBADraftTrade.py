
# coding: utf-8

# # Draft Pick - Draft Pick

# In[1]:


# Some useful libraries - also allowing the entire dataframe to get printed out
import pandas as pd
import numpy as np
import math 
import seaborn as sns
from scipy import stats 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


pd.set_option('display.max_rows', 5000)


# In[2]:


# Cleaning up the data - no draft picks outside the 2nd round, no players without stats
# Data from basketball-reference.com - thanks!

draft = pd.read_csv('NBA_Draft_data.csv')
draft = draft[draft['Rd'] < 3]
draft = draft[pd.isnull(draft['WS/48']) == False]
draft = draft[draft['Pk'] < 61]
draft['Age'] = round(draft['Age'])


# In[3]:


# Let's take a look at the pick number vs Win Shares per 48 minutes

pick_ws48 = draft.groupby(['Pk'])['WS/48'].mean()
pick_ws48 = pd.DataFrame(pick_ws48)
pick_ws48['Pick'] = pick_ws48.index

ax = sns.regplot(x="Pick", y="WS/48", data=pick_ws48)
ax.set(xlabel='Draft Pick #', ylabel='Win Shares per 48 Minutes')
ax.set_title("Draft Pick # vs WS Per 48 Mins")


# In[4]:


# Let's take a look at the pick number vs Win Shares Total

pick_ws = draft.groupby(['Pk'])['WS'].mean()
pick_ws = pd.DataFrame(pick_ws)
pick_ws['Pick'] = pick_ws.index


# In[5]:


def func(x, a, b, c):
    return a * np.exp(-b * x) + c

popt, pcov = curve_fit(func, pick_ws['Pick'], pick_ws['WS'])

ax = sns.scatterplot(x="Pick", y="WS", data=pick_ws)
ax.set(xlabel='Draft Pick #', ylabel='Win Shares')
ax.set_title("Draft Pick # vs Total Win Shares")
plt.plot(pick_ws['Pick'], func(pick_ws['Pick'], *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

# R squared calculations
residuals = pick_ws['WS'] - func(pick_ws['Pick'], 58.55173513, 0.08811615, 6.73865641)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((pick_ws['WS']-np.mean(pick_ws['WS']))**2)
r_squared = 1 - (ss_res / ss_tot)
r_squared
# R-squared is 0.9056497630031183, very good


# In[6]:


# Smoothen WS values
pick_ws['SmoothWS'] = (popt[0] * np.exp(-popt[1] * pick_ws['Pick'])) + popt[2]


# In[7]:


# For a pick #, what are the average total WS and WS/48

def giveAvgWS(pick):
    return round(pick_ws[pick_ws['Pick'] == pick]['SmoothWS'].mean(), 2)

def giveAvgWS48(pick):
    return round(pick_ws[pick_ws['Pick'] == pick]['WS/48'].mean(), 2)


# In[8]:


# Let's make some evaluations
# IF THE RESULT IS NEGATIVE, TEAM 1 WINS THE TRADE

def evalTradeWS(team1picks, team2picks):
    team1WS = 0
    for pick in team1picks:
        team1WS += giveAvgWS(pick)
    
    team2WS = 0
    for pick in team2picks:
        team2WS += giveAvgWS(pick)
        
    if team1WS > team2WS:
        return team1WS - team2WS
    
    else:
        return team1WS - team2WS
    
def whoWins(team1picks, team2picks):
    val = evalTradeWS(team1picks, team2picks)
    if val < 0:
        print('team 1 wins')
    else:
        print('team 2 wins')


# In[9]:


whoWins([15, 16], [14, 17])


# In[10]:


# This isn't that useful  
def evalTradeWS48(team1picks, team2picks):
    team1WS = 0
    for pick in team1picks:
        team1WS += giveAvgWS48(pick)
    
    team2WS = 0
    for pick in team2picks:
        team2WS += giveAvgWS48(pick)
        
    if team1WS > team2WS:
        return team1WS - team2WS
    
    else:
        return team1WS - team2WS


# In[11]:


# Let's try to make the trades fair
def makeMoreFair(team1picks, team2picks): 
    # Get the difference in the trade
    diff = evalTradeWS(team1picks, team2picks)
    team1Wins = True
    team2Wins = True
    
    # If the difference is positive, team 2 loses the trade
    if diff > 0: 
        team1Wins = False

    else:
        team2Wins = False
        
    # Say team 1 is winning the trade
    if team1Wins == True: 
        # Loop through the rows
        shouldBreak = False
        for index, row in pick_ws[::-1].iterrows():
            pickAdd = row['Pick']
            if pickAdd not in team1picks:
                team1picks.append(pickAdd)
                team1picks = list(dict.fromkeys(team1picks))
                newDiff = (evalTradeWS(team1picks, team2picks))
                # Is the new difference better? 
                if newDiff > 0 and pickAdd != 60:
                    print('Give team two the', pickAdd + 1, 'pick')
                    shouldBreak = True
                    
                elif newDiff > 0 and pickAdd == 60:
                    print('This is the best deal possible')
                    shouldBreak = True
                
                else:
                    team1picks.remove(pickAdd)    
                    
            if shouldBreak == True:
                break
    elif team2Wins == True:
        # Loop through the rows
        shouldBreak = False
        for index, row in pick_ws[::-1].iterrows():
            pickAdd = row['Pick']
            if pickAdd not in team2picks:
                team2picks.append(pickAdd)
                team2picks = list(dict.fromkeys(team2picks))
                newDiff = (evalTradeWS(team1picks, team2picks))
                # Is the new difference better? 
                if newDiff < 0 and pickAdd != 60:
                    print('Give team one the', pickAdd + 1, 'pick')
                    shouldBreak = True
                elif newDiff < 0 and pickAdd == 60:
                    print('This is the best deal possible')
                    shouldBreak = True
                else:
                    team2picks.remove(pickAdd)    
                
            if shouldBreak == True:
                break
        


# In[12]:


# Here is an unfair trade where team 1 is getting ripped off - the number reported is the 
# difference in win shares between the two pick packages
evalTradeWS([8], [15, 41])


# In[13]:


# Let's make this more fair!
makeMoreFair([1], [60])


# In[14]:


# Now here's the new trade's difference in win shares
evalTradeWS([8], [15, 41, 60])


# # Player - Draft Pick

# In[15]:


# Read in the data
winShareData = pd.read_csv('WinShareData.csv')
minAge = winShareData['Age'].min()
maxAge = winShareData['Age'].max()


# In[16]:


# For every age, find the average amount of Win Shares

wsByYear = []
for i in range(minAge,maxAge+1):
    tframe = winShareData[winShareData['Age'] == i]
    tradedPlayers = list(tframe[tframe['Tm'] == 'TOT']['Player'])

    for index, row in tframe.iterrows():
        if row['Player'] in tradedPlayers and row['Tm'] != 'TOT':
            tframe.drop(index, inplace=True)
            
    avgWS = tframe['WS'].mean()
    wsByYear.append(avgWS)


# In[17]:


# Taking a look at Player age vs WS

wsby = pd.DataFrame(wsByYear)
wsby['age'] = wsby.index
wsby['age'] = wsby['age'] + 19
wsby.columns = ['WS', 'Age']

ax = sns.scatterplot(x="Age", y="WS", data=wsby)
ax.set(xlabel='Player Age', ylabel='Win Shares (Yearly)')
ax.set_title("Player Age  vs WS")


# In[23]:


# Employ a quadratic regression to smooth win share vs age data

import operator

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

x = wsby['Age'].values.reshape(-1, 1)
y = wsby['WS']

polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)
# R-squared is 0.8705593525409101, pretty good

plt.scatter(x, y, s=10)
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='m')
plt.title("Player Age  vs WS")
plt.show()

y_poly_pred = list(y_poly_pred)
smoothWSAge = pd.DataFrame(y_poly_pred)

smoothWSAge['Age'] = smoothWSAge.index
smoothWSAge['Age'] = smoothWSAge['Age'] + 19
smoothWSAge.columns = ['WS', 'Age']


# In[24]:


# Get the ratio of current WS vs expected future WS
def getWSRatio(age, futureage):
    currentWS = smoothWSAge.loc[smoothWSAge['Age'] == age]['WS'].values[0]
    futureWS = smoothWSAge.loc[smoothWSAge['Age'] == futureage]['WS'].values[0]
    ratio = futureage / age
    return ratio


# In[25]:


# Let's get the expected WS over the course of a player contract
def playerContractWS(prevYrWS, age, contractLength):
    totalWS = 0
    AGE = age
    for i in range(age + 1, age + 1 + contractLength):
        currRatio = getWSRatio(AGE, AGE + 1)
        totalWS += (prevYrWS * currRatio)
        prevYrWS = (prevYrWS * currRatio)
        AGE += 1
        
    return totalWS


# In[27]:


# Rookie WS based on 4 year contract
# Pct of WS over a career earned on a rookie contract
wsby1 = wsby[wsby['Age'] > 19]
rookieWSPct = sum(wsby1[wsby1['Age'] < 24]['WS'].values) / sum(wsby['WS'].values)

def rookieWS(pickno):
    WS = giveAvgWS(pickno)
    scale = WS * rookieWSPct
    return scale


# In[28]:


# Draft pick vs Current Players
def evalTradeWSP2P(players, picks):
    playerWS = 0
    for player in players:
        playerWS += playerContractWS(player[0], player[1], player[2])
    
    picksWS = 0
    for pick in picks:
        picksWS += rookieWS(pick)
        
    if playerWS > picksWS:
        return playerWS - picksWS
    
    else:
        return playerWS - picksWS
    
def whoWinsP2P(players, picks):
    val = evalTradeWSP2P(players, picks)
    if val < 0:
        print('Picks worth more')
    else:
        print('Player(s) worth more')


# In[108]:


def makeMoreFairP2P(players, picks): 
    # Get the difference in the trade
    diff = evalTradeWSP2P(players, picks)
    playersWin = True
    picksWin = True
    # If the difference is positive, team 2 loses the trade
    if diff > 0: 
        playersWin = False

    else:
        picksWin = False
            
    # Say picks are winning the trade
    if picksWin == True: 
        # Loop through the rows
        shouldBreak = False
        for index, row in pick_ws[::-1].iterrows():
            pickAdd = row['Pick']
            if pickAdd not in picks:
                picks.append(pickAdd)
                picks = list(dict.fromkeys(picks))
                newDiff = (evalTradeWSP2P(players, picks))
                # Is the new difference better? 
                if newDiff < 0 and pickAdd != 60:
                    print('Give Player Team the', pickAdd + 1, 'pick')
                    shouldBreak = True
                    
                elif newDiff < 0 and pickAdd == 60:
                    print('This is the best deal possible')
                    shouldBreak = True
                
                else:
                    picks.remove(pickAdd)    
                    
            if shouldBreak == True:
                break
                
        if shouldBreak == False: 
            print('Give Player Team the no. 1 overall pick')
            
        
    elif playersWin == True:
        # Loop through the rows
        shouldBreak = False
        for index, row in pick_ws[::-1].iterrows():
            pickAdd = row['Pick']

            if pickAdd != 500:
                pickAddVal = rookieWS(pickAdd)
                newDiff = (evalTradeWSP2P(players, picks)) + pickAddVal
                # Is the new difference better? 
                if newDiff > 0 and pickAdd != 60:
                    print('Give the player team the', pickAdd + 1, 'pick')
                    shouldBreak = True
                elif newDiff > 0 and pickAdd == 60:
                    print('This is the best deal possible')
                    shouldBreak = True
                # else:
                #     team2picks.remove(pickAdd)    
                
            if shouldBreak == True:
                break


# In[113]:


player = [5, 22, 4]
makeMoreFairP2P([player], [1,1, 1])

