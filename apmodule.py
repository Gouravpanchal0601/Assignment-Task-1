import numpy as np
import pandas as pd
import scipy.optimize as sco
from scipy import stats

"""
This is the second version of our Applied Portfolio Management Library where
I have stored some of the functions we have developed together so far.

It is always useful to add some comments that  maybe useful to the user of the
library
"""
####################################
#       Portfolio Strategies       #
####################################



# Returns the Market Cap weighted portfolio in the top N stocks
def rank_strategy(factor_data, market_cap, N=100):
    
    #We rank the shares
    rank = factor_data.groupby('date').rank(ascending=False, method='min')
    
    #We calculate the signal
    signal = (rank <= N).astype(int)
    
    #We join signal and market cap
    df = signal.rename('signal').to_frame().join(market_cap)
    
    #We calculate the position
    position = df['market_cap'] / df.groupby(['date', 'signal'])['market_cap'].transform('sum')
    position = position * df['signal']
    position = position.rename('position')
    
    #We calculate the benchmark as the market weighted portfolio with all the shares
    benchmark = df['market_cap'] / df.groupby('date')['market_cap'].transform('sum')
    
    #We return the two series
    return position, benchmark


# Returns the Market Cap weighted portfolio long in the top N stocks and short in the bottom N
def mn_strategy(factor_data, market_cap, N=100):
    
    #We build the long signal
    rank = factor_data.groupby('date').rank(ascending=False, method='min')
    signal_long = (rank <= N).astype(int)
    
    #We build the short signal
    rank = factor_data.groupby('date').rank(ascending=True, method='min')
    signal_short = (rank <= N).astype(int)
    
    #We build the final signal
    signal = signal_long - signal_short
    
    #We join signal and market cap
    df = signal.rename('signal').to_frame().join(market_cap)
    
    #We calculate the position
    position = df['market_cap'] / df.groupby(['date', 'signal'])['market_cap'].transform('sum')
    position = position * df['signal']
    position = position.rename('position')
    
    return position

# Returns the Market Cap weighted portfolio overinvested in the top N stocks and underinvested in the bottom N
def ls_strategy(factor_data, market_cap, N=100, active=0.3):
    
    #We build the long signal
    rank = factor_data.groupby('date').rank(ascending=False, method='min')
    signal_long = (rank <= N).astype(int)
    
    #We build the short signal
    rank = factor_data.groupby('date').rank(ascending=True, method='min')
    signal_short = (rank <= N).astype(int)
    
    #We build the final signal
    signal = signal_long - signal_short
    
    #We join signal and market cap
    df = signal.rename('signal').to_frame().join(market_cap)
    
    #We calculate the weight of the stocks in the market
    market_weight = df['market_cap'] / df.groupby('date')['market_cap'].transform('sum')
    
    #We calculate the active weight in the top/bottom N stocks
    active_weight = df['signal'] * active / N
    
    #We calculate the position as the sum of market and active weight
    position = market_weight + active_weight
    position = position.rename('position')
    
    #We calculate the benchmark as the market weighted portfolio with all the shares
    benchmark = df['market_cap'] / df.groupby('date')['market_cap'].transform('sum')
    benchmark = benchmark.rename('benchmark')
    
    #We return the two series
    return position, benchmark

# Calculate standard diagnostics
def diagnostics(port_ret):
    mean_return = port_ret.mean()*12
    volatility = port_ret.std()*np.sqrt(12)
    rr_ratio = (port_ret.mean()*12) / (port_ret.std()*np.sqrt(12))
    p_positive = port_ret[port_ret>0].count()/port_ret.count()
    worst = port_ret.min()
    best = port_ret.max()
    
    #we derive the portfolio values from the monthly returns
    port_val = port_ret.cumsum().apply(np.exp)
    rolling_peak = port_val.rolling(12).max()
    drawdown = np.log(port_val/rolling_peak)
    max_drawdown = drawdown.min().rename('Max Drawdown')

    performance = pd.DataFrame({'Mean Return': mean_return,
                                'St. Dev.': volatility,
                                'RR Ratio': rr_ratio,
                                '% Positive': p_positive,
                                'Worst Month': worst,
                                'Best Month': best,
                                'Max DrawDown': max_drawdown})
    
    return performance.transpose()



####################################
#       Backtesting  Fuctions      #
####################################

# Backtesting of a long-only strategy. Returns monthly returns
def backtesting(info_signal, prices, market_cap, start='1900-01-01', end='2100-12-31', frequency=1, t_cost=0, N=100):
    
    position, benchmark = rank_strategy(info_signal, market_cap, N)
    
    #we derive the future returns
    future_returns = np.log(prices.groupby('id').shift(-1) / prices).rename('fut_ret')
    
    #we limit the sample to the period requested
    position = position.loc[pd.IndexSlice[:, start:end]]
    benchmark = benchmark.loc[pd.IndexSlice[:, start:end]]

    #we build the list of rebalancing dates 
    rebalance = pd.Series(0, index=position.index.get_level_values('date').unique()).sort_index()
    rebalance.iloc[::frequency] = 1
    
    #we build the empty initial portfolio
    current_weight = pd.Series(0, index=position.index.get_level_values('id').unique()).rename('current_weight').sort_index()
    
    #we create the empty list for the results of the backtesting
    result = []
    
    #we start the loop
    for row in rebalance.items():

        #STEP 1: We isolate the portfolio weights and the returns relative to the specific month
        p = position.loc[pd.IndexSlice[:, row[0]]]
        r = future_returns.loc[pd.IndexSlice[:, row[0]]]

        #STEP 2: We check if the month is a rebalancing month
        if row[1] == 1:
            #we attach the target weights to the current weights
            port=current_weight.to_frame().join(p, how='left').fillna(0)
            #we calculate the turnover
            turnover = 0.5*(port['position'] - port['current_weight']).abs().sum()
            #we update the weights
            current_weight = port['position'].rename('current_weight')
        else:
            #if no rebalancing then turnover is zero
            turnover = 0

        #STEP 3: We calculate portfolio return
        port=current_weight.to_frame().join(r, how='left').fillna(0)
        gross_performance = (port['current_weight'] * port['fut_ret']).sum()

        #we adjust for transaction costs
        tcost = turnover * t_cost
        net_performance = gross_performance - tcost

        #STEP 4: We adjust the portfolio weights 
        new_weight = port['current_weight'] * (1 + port['fut_ret'])
        new_weight = new_weight / new_weight.sum()    
        current_weight = new_weight.rename('current_weight')

        #we export the results by appending them in the result list
        result.append([row[0], net_performance, turnover, tcost])
    
    #We calculate the benchmark return
    ben_ret= benchmark.rename('weight').to_frame().join(future_returns)
    ben_ret = ben_ret['weight']*ben_ret['fut_ret']
    ben_ret = ben_ret.groupby('date').sum().rename('Benchmark')
    
    #We calculate the active return (portfolio minus benchmark)
    port_ret = pd.DataFrame(result, columns=['date', 'Portfolio', 'Turnover', 'T-Cost'])
    port_ret = port_ret.set_index('date')
    port_ret = ben_ret.to_frame().join(port_ret)
    port_ret['Active'] = port_ret['Portfolio'] - port_ret['Benchmark']
    
    #We prepare the output objects
    monthly_returns = port_ret[['Portfolio', 'Benchmark', 'Active']]
    turnover = port_ret[['Turnover', 'T-Cost']]
    composition = position.rename('Portfolio').to_frame().join(benchmark.rename('Benchmark'))
    performance = diagnostics(monthly_returns)
        
    #we return the output DataFrame with the dates set as index
    return monthly_returns, turnover, composition, performance


# Backtesting of a market-neutral strategy. Returns monthly returns
def mn_backtesting(info_signal, prices, market_cap, start='1900-01-01', end='2100-12-31', frequency=1, t_cost=0, N=100):
    
    position = mn_strategy(info_signal, market_cap, N)
    
    #we derive the future returns
    future_returns = np.log(prices.groupby('id').shift(-1) / prices).rename('fut_ret')
    
    #we limit the sample to the period requested
    position = position.loc[pd.IndexSlice[:, start:end]]

    #we build the list of rebalancing dates 
    rebalance = pd.Series(0, index=position.index.get_level_values('date').unique()).sort_index()
    rebalance.iloc[::frequency] = 1
    
    #we build the empty initial portfolio
    current_weight = pd.Series(0, index=position.index.get_level_values('id').unique()).rename('current_weight').sort_index()
    
    #we create the empty list for the results of the backtesting
    result = []
    
    #we start the loop
    for row in rebalance.items():

        #STEP 1: We isolate the portfolio weights and the returns relative to the specific month
        p = position.loc[pd.IndexSlice[:, row[0]]]
        r = future_returns.loc[pd.IndexSlice[:, row[0]]]

        #STEP 2: We check if the month is a rebalancing month
        if row[1] == 1:
            #we attach the target weights to the current weights
            port=current_weight.to_frame().join(p, how='left').fillna(0)
            #we calculate the turnover
            turnover = 0.5*(port['position'] - port['current_weight']).abs().sum()
            #we update the weights
            current_weight = port['position'].rename('current_weight')
        else:
            #if no rebalancing then turnover is zero
            turnover = 0

        #STEP 3: We calculate portfolio return
        port=current_weight.to_frame().join(r, how='left').fillna(0)
        gross_performance = (port['current_weight'] * port['fut_ret']).sum()

        #we adjust for transaction costs
        tcost = turnover * t_cost
        net_performance = gross_performance - tcost

        #STEP 4: We adjust the portfolio weights 
        new_weight = port['current_weight'] * (1 + port['fut_ret'])
        new_weight[new_weight>0] = new_weight[new_weight>0] / new_weight[new_weight>0].sum()
        new_weight[new_weight<0] = new_weight[new_weight<0] / new_weight[new_weight<0].abs().sum()
        current_weight = new_weight.rename('current_weight')

        #we export the results by appending them in the result list
        result.append([row[0], net_performance, turnover, tcost])
    
    
    #We prepare the output objects
    port_ret = pd.DataFrame(result, columns=['date', 'MN Portfolio', 'Turnover', 'T-Cost'])
    port_ret = port_ret.set_index('date')
    monthly_returns = port_ret['MN Portfolio'].to_frame()
    turnover = port_ret[['Turnover', 'T-Cost']]
    composition = position.rename('MN Portfolio')
    performance = diagnostics(monthly_returns)
        
    #we return the output DataFrame with the dates set as index
    return monthly_returns, turnover, composition, performance


# Backtesting of long-short strategy in the spirit of a 130/30 portfolio. Returns monthly returns
def ls_backtesting(info_signal, prices, market_cap, start='1900-01-01', end='2100-12-31', frequency=1, t_cost=0, N=100, active = 0.3):
    
    position, benchmark = ls_strategy(info_signal, market_cap, N, active)
    
    #we derive the future returns
    future_returns = np.log(prices.groupby('id').shift(-1) / prices).rename('fut_ret')
    
    #we limit the sample to the period requested
    position = position.loc[pd.IndexSlice[:, start:end]]
    benchmark = benchmark.loc[pd.IndexSlice[:, start:end]]

    #we build the list of rebalancing dates 
    rebalance = pd.Series(0, index=position.index.get_level_values('date').unique()).sort_index()
    rebalance.iloc[::frequency] = 1
    
    #we build the empty initial portfolio
    current_weight = pd.Series(0, index=position.index.get_level_values('id').unique()).rename('current_weight').sort_index()
    
    #we create the empty list for the results of the backtesting
    result = []
    
    #we start the loop
    for row in rebalance.items():

        #STEP 1: We isolate the portfolio weights and the returns relative to the specific month
        p = position.loc[pd.IndexSlice[:, row[0]]]
        r = future_returns.loc[pd.IndexSlice[:, row[0]]]

        #STEP 2: We check if the month is a rebalancing month
        if row[1] == 1:
            #we attach the target weights to the current weights
            port=current_weight.to_frame().join(p, how='left').fillna(0)
            #we calculate the turnover
            turnover = 0.5*(port['position'] - port['current_weight']).abs().sum()
            #we update the weights
            current_weight = port['position'].rename('current_weight')
        else:
            #if no rebalancing then turnover is zero
            turnover = 0

        #STEP 3: We calculate portfolio return
        port=current_weight.to_frame().join(r, how='left').fillna(0)
        gross_performance = (port['current_weight'] * port['fut_ret']).sum()

        #we adjust for transaction costs
        tcost = turnover * t_cost
        net_performance = gross_performance - tcost

        #STEP 4: We adjust the portfolio weights 
        new_weight = port['current_weight'] * (1 + port['fut_ret'])
        new_weight = new_weight / new_weight.sum()    
        current_weight = new_weight.rename('current_weight')

        #we export the results by appending them in the result list
        result.append([row[0], net_performance, turnover, tcost])
    
    #We calculate the benchmark return
    ben_ret= benchmark.rename('weight').to_frame().join(future_returns)
    ben_ret = ben_ret['weight']*ben_ret['fut_ret']
    ben_ret = ben_ret.groupby('date').sum().rename('Benchmark')
    
    #We calculate the active return (portfolio minus benchmark)
    port_ret = pd.DataFrame(result, columns=['date', 'LS Portfolio', 'Turnover', 'T-Cost'])
    port_ret = port_ret.set_index('date')
    port_ret = ben_ret.to_frame().join(port_ret)
    port_ret['LS Active'] = port_ret['LS Portfolio'] - port_ret['Benchmark']
    
    #We prepare the output objects
    monthly_returns = port_ret[['LS Portfolio', 'Benchmark', 'LS Active']]
    turnover = port_ret[['Turnover', 'T-Cost']]
    composition = position.rename('LS Portfolio').to_frame().join(benchmark.rename('Benchmark'))
    performance = diagnostics(monthly_returns)
        
    #we return the output DataFrame with the dates set as index
    return monthly_returns, turnover, composition, performance


######################################################
#   Fuctions that return optimal portfolio weights   #
######################################################

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights.T, mean_returns)  
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return std, returns

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_std, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_std

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    
    #Extract the number of assets from the length of the returns vector
    num_assets = len(mean_returns)
    
    #Creates a tuple with the variables to be uses by the objective function
    args = (mean_returns, cov_matrix, risk_free_rate)
    
    #The Constrating that the sum of the weight has to be equal to one
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Individual weights between minus 100% and plus 100%. We allow for negative 
    # weights in case we have a factor that performs really poorly we may add it 
    # with negative weight
    bound = (0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    
    # This runs the actual optimization
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    
    # This put the results in a series
    opt_weights = pd.Series(result['x'], index=mean_returns.index)
    return opt_weights

def optimized_alpha_model(active_returns):
    
    #We calculate the optimization inputs
    mean_returns = active_returns.mean()*12
    cov_matrix = active_returns.cov()*12
    
    #We run the optimization
    opt_weights = max_sharpe_ratio(mean_returns, cov_matrix, 0)
    
    #We return the optimal weights
    return opt_weights

def walkforward_alpha_model(active, T):
    
    #We create an empty DataFrame with the same columns as the input
    # model = pd.DataFrame(columns=active.columns)
    result = []
    
    #We take the number of rows in the input
    n_rows = active.shape[0]
    
    #We start the loop
    for n in range(n_rows - T + 1):
        
        #We take a portion of the input from row n to n+T
        active_t = active.iloc[n:n+T,:]
        
        #We run the factor optimization function
        weights = optimized_alpha_model(active_t)
        
        #We appent the results to the output DataFrame
        # model = model.append(weights,ignore_index=True)
        result.append(weights)
    
    #We create a DataFrame using the input column names and
    #using a portion of the index of the input as the index of the output
    model = pd.DataFrame(result, columns=active.columns).set_index(active.index[T-1:])
        
    #We return the output
    return model


################################################################
#   Fuctions that perform non-linear predictability analysis   #
################################################################

def ic_analysis(signal, prices, frequency='monthly'):
    
    #We load all the modules needed in the function just in case they are not loaded in the notebook.
    #They are only active within the function, if you need them elsewhere in the notebook you need to 
    #import them again in the notebook
    import pandas as pd
    import numpy as np
    from scipy import stats
    
    #We check that the frequency is correct. If not we stop the function and return an empty result
    if frequency not in ['monthly', 'quarterly', 'annual']:
        print('Warning, the frequency must be monthly, quarterly or annual')
        return None
    
    #We capture the name of the factor using the .name attribute to use it later
    signal_name = signal.name
    
    #We calculate the returns according to the periodicity chosen in the input
    if frequency=='monthly':
        future_returns = np.log(prices.groupby('id').shift(-1) / prices).rename('fut_ret')
        
    elif frequency=='quarterly':
        future_returns = np.log(prices.groupby('id').shift(-3) / prices).rename('fut_ret')
        
    elif frequency=='annual':
        future_returns = np.log(prices.groupby('id').shift(-12) / prices).rename('fut_ret')
    
        
    #We join the signal with the future return
    data = signal.to_frame().join(future_returns).dropna()
    data['month'] = data.index.get_level_values('date').month
    
    #We select the relevant months for the annual or quarterly analysis
    if frequency=='annual':
        data = data[data['month']==12]
        
    elif frequency=='quarterly':
        data = data[data['month'].isin([3, 6, 9, 12])] #.isin() checks if the number is among the ones in the list
    
    #We drop the month column because we do not need it anymore
    data.drop(columns=['month'], inplace=True)
    
    #We calculate the IC
    ic = data.groupby('date').corr(method='spearman').iloc[0::2,-1].droplevel(level=1).rename('IC')
    
    #We print on screen the average IC
    print('Average IC:', round(ic.mean(), 3), '\n')
    
    #We calculate and print the percentage of positive/negative observations
    sign = np.sign(ic).value_counts() / ic.count()
    
    print('Percentage of Positive Periods:', round(sign.loc[1], 3), '\n')
    print('Percentage of Negative Periods:', round(sign.loc[-1], 3), '\n')
    
    #We calculate and print the t-test
    t_test = stats.ttest_1samp(ic, 0)
    print(f'T-Stat: {round(t_test.statistic,3)} P-Value: {round(t_test.pvalue,3)} \n')
    
    #We return the series with the Information Coefficient for further analysis using the original name of the factor
    return ic.rename('IC_' + signal_name)


def quantile_analysis(signal, prices, n_bins=4):
    
    #We capture the name of the factor using the .name attribute to use it later
    signal_name = signal.name
    
    #We calculate the future returns
    future_returns = np.log(prices.groupby('id').shift(-1) / prices).rename('fut_ret')
    
    #We join the signal with the future returns
    data = signal.to_frame().join(future_returns).dropna()
    
    #We apply the pandas.qcut() function using a lambda function
    data['group'] = data.groupby('date', group_keys=False)[signal_name].apply(lambda x: pd.qcut(x,n_bins, labels=False))+1
    
    #We calculate the average return of the groups
    portfolios = data.groupby(['date','group']).mean()
    
    #We rearrange the results in order to have one group per column
    port_returns = portfolios['fut_ret'].unstack(level=1) 
    
    #We calculate the benchmark return
    benchmark_return = data.groupby('date')['fut_ret'].mean().rename('Benchmark')
    port_returns = port_returns.join(benchmark_return.to_frame())
    
    #We calculate the return of the zero-investment portfolios
    port_returns['Active'] = port_returns[n_bins] - port_returns['Benchmark']
    port_returns['Neutral'] = port_returns[n_bins] - port_returns[1]
    
    performance = diagnostics(port_returns)
    
    return port_returns, performance


#####################################################################
#   Fuctions that analyse returns of machine learning predictions   #
#####################################################################

def ml_analysis(prediction, prices):
    
    #We import the apmodule in case it is not imported in the notebook
    import apmodule as ap
    
    #We capture the name of the factor using the .name attribute to use it later
    pred_name = prediction.name
    
    #We look for the tag of the best and worst outcome
    best = prediction.max()
    worst = prediction.min()
    
    #We calculate the future returns
    future_returns = np.log(prices.groupby('id').shift(-1) / prices).rename('fut_ret')
    
    #We join the signal with the future returns
    data = prediction.to_frame().join(future_returns).dropna()
    
    #We calculate the average return of the groups
    portfolios = data.groupby(['date',pred_name]).mean()
    
    #We rearrange the results in order to have one group per column
    port_returns = portfolios['fut_ret'].unstack(level=1) 
    
    #We calculate the benchmark return
    benchmark_return = data.groupby('date')['fut_ret'].mean().rename('Benchmark')
    port_returns = port_returns.join(benchmark_return.to_frame())
    
    #We calculate the return of the zero-investment portfolios
    port_returns['Active'] = port_returns[best] - port_returns['Benchmark']
    port_returns['Neutral'] = port_returns[best] - port_returns[worst]
    
    diagnostics = ap.diagnostics(port_returns)

    return port_returns, diagnostics