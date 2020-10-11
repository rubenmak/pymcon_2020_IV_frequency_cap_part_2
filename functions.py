import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def generate_simulation_data(P_potential_impressions,
                             P_frequency_cap,
                             base_conversion_probability,
                             effects,
                             n_users):
    max_cap = len(P_frequency_cap)
    users_per_frequency_cap = stats.multinomial.rvs(n_users,
                                                    P_frequency_cap)
    users = []
    frequency_cap = []
    impressions = []
    conversions = []
    for i, u in enumerate(users_per_frequency_cap):
        us, f_cap, impr, conv = generate_data_per_frequency_cap(u,
                                                                i + 1,
                                                                P_potential_impressions,
                                                                base_conversion_probability,
                                                                effects)
        users += us
        frequency_cap += f_cap
        impressions += impr
        conversions += conv

    df = pd.DataFrame({'users': users,
                       'frequency_cap': frequency_cap,
                       'impressions': impressions,
                       'conversions': conversions})
    df = df.groupby(['frequency_cap', 'impressions']).agg({'users': 'sum', 'conversions': 'sum'})
    return df


def generate_data_per_frequency_cap(n_users,
                                    frequency_cap,
                                    P_potential_impressions,
                                    base_conversion_probability,
                                    effects):
    max_impr = len(P_potential_impressions)
    users = list(stats.multinomial.rvs(n_users, P_potential_impressions))
    frequency_cap_array = [frequency_cap] * max_impr
    impressions = [min(i + 1, frequency_cap) for i in range(max_impr)]
    total_effects = np.array([np.sum(effects[:i]) for i in impressions])
    P_conversion = np.array(base_conversion_probability) + total_effects
    conversions = [stats.binom.rvs(users[i], P_conversion[i]) for i in range(len(users))]
    return users, frequency_cap_array, impressions, conversions

def plot_posteriors(x, posterior_samples, true_values = None, percentiles = [0.03, 0.97]):
    if true_values:
        plt.plot(x, true_values, label='true values')
    mean_y = [np.mean(s) for s in posterior_samples]
    plt.plot(x, mean_y, label='mean')
    for p in percentiles:
        percentile_y = [np.sort(s)[round(p * len(s))] for s in posterior_samples]
        plt.plot(x, percentile_y, label='{}.% percentile'.format(p*100))
    plt.legend()
    plt.show()