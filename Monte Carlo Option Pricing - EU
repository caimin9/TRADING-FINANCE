import numpy as np

def monte_carlo_option_price_complex(r, sigma, S_0, K, T, num_simulations, num_steps):
    """
    r: risk-free rate
    sigma: volatility of underlying asset
    S_0: initial stock price
    K: strike price
    T: time to maturity
    num_simulations: number of simulations
    num_steps: number of time steps in the simulation
    """
    dt = T / num_steps
    stock_prices = np.zeros((num_steps + 1, num_simulations))
    stock_prices[0] = S_0

    # Simulating the path of stock price
    for t in range(1, num_steps + 1):
        brownian = np.random.standard_normal(num_simulations)
        stock_prices[t] = stock_prices[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * brownian)

    # Payoff from a European call option
    payoff = np.maximum(stock_prices[-1] - K, 0)

    # Discounted expected payoff
    option_price = np.exp(-r * T) * np.sum(payoff) / num_simulations

    return option_price
