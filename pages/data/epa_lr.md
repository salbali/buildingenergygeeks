---
title: Linear regression
tags: [assessment, linear]
keywords: linear regression, building, energy, performance, assessment
last_updated: March 29, 2020
summary: "Under some strong hypotheses, the energy balance of a building can be approximated by linear functions. Despite their limitations, linear regression models can be very useful as a first insight into the heat balance of a building."
sidebar: sidebar_data
topnav: topnav_data
permalink: epa_lr.html
#folder: mydoc
---


Linear regression models are one of the most simple forms of relationship that can be proposed between measured data. They often serve as an introduction to statistical learning ([see this book as a great example](https://faculty.marshall.usc.edu/gareth-james/ISL/)) because they offer a simple framework to demonstrate the important steps that a data analyst should follow: model selection, hypothesis testing, validation...

Under some strong hypotheses, the energy balance of a building can be approximated by linear functions. They however have several limitations: they cannot represent non-linear phenomena, such as radiative heat exchange between walls; they don't allow identifying the parameters driving dynamical phenomena; they impose a fixed structure to the energy balance equation.

Despite these limitations, linear regression models can however be very useful as a first insight into the heat balance of a building: they allow a quick assessment of which types of measurements have an impact on the global balance and guide the choice of more detailed models. Moreover, if a large enough amount of data is available, the estimates of some coefficients such as the HLC often turn out to be quite reliable.

In the following, we show a practical application of linear regression to the identification of the important phenomena that influence the energy use in a house.


## The data

The data used in this example was published by the Oak Ridge National Laboratory, Building Technologies Research and Integration Center (USA). It contains end use breakdowns of energy use and various indoor environmental conditions collected at the Campbell Creek Research House 3, at a 15 minute time stamp. The data availability ranges from 10/1/2013 to 9/30/2014. It is available for download [here](https://openei.org/datasets/dataset/ornl-research-house-3).

This dataset was chosen in this example for the diversity and duration of the available measurements.

Before taking a look at the data, let us start with some imports.

```python
# The holy trinity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# What we will use for regression
import statsmodels.api as sm

# bokeh is very good for a first exploration of a dataset
from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.palettes import Category10
from bokeh.io import output_notebook
output_notebook()
```

```python
# Opening the data file and showing the timestamps to pandas
df = pd.read_excel('data/ornlbtricdatafromcc3fy2014.xlsx', header=1).iloc[2:]
df.set_index(pd.to_datetime(df['TIMESTAMP']), inplace=True, drop=True)

# Dealing with missing values
df.replace('NAN', np.nan, inplace=True)
df.fillna(method='pad', inplace=True)

df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>TIMESTAMP</th>
      <th>RECORD</th>
      <th>main_Tot</th>
      <th>Batt_Volt_Avg</th>
      <th>PV_generated_Tot</th>
      <th>HP_in_Tot</th>
      <th>...</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-10-01 00:15:00</th>
      <td>767</td>
      <td>175.0</td>
      <td>12.82</td>
      <td>0.00</td>
      <td>2.50</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2013-10-01 00:30:00</th>
      <td>768</td>
      <td>250.0</td>
      <td>12.81</td>
      <td>0.75</td>
      <td>5.00</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2013-10-01 00:45:00</th>
      <td>769</td>
      <td>182.5</td>
      <td>12.81</td>
      <td>0.75</td>
      <td>5.00</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2013-10-01 01:00:00</th>
      <td>770</td>
      <td>165.0</td>
      <td>12.81</td>
      <td>0.75</td>
      <td>6.25</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2013-10-01 01:15:00</th>
      <td>771</td>
      <td>347.5</td>
      <td>12.81</td>
      <td>0.00</td>
      <td>5.00</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 103 columns</p>
</div>


This dataframe has quite a lot of features (103 columns), as shown by the output of the `.head()` method. The meaning of each column is specified in a separate table. This is where I recommend using the `bokeh` library for its convenient plotting tools, shown as an example below.

The house is heated and cooled by a heat pump, and most energy uses (plugs, appliances, hot water production...) are broken down and measured separately. Temperatures, relative humidities are also available in several locations. We are going to use the **energy use of the heat pump** as the output variable of linear regression models, and try to explain it with the other available variables.

In the following block, a new dataframe is created to only keep variables that we believe may influence the heating and cooling energy consumption of the house. Note that this is already an important decision in the data analysis process, as we might be filtering out information that could have been useful.
* Energy readings: heat pump, domestic hot water production, ventilation fan power and other uses.
* Temperatures: average indoor temperature, garage (adjacent unheated space), ventilation supply temperature and outdoor temperature.
* Weather variables: solar irradiance and wind speed are usually known to impact the energy balance.

Finally, only the months of November to March are kept in this exercise.


```python
df2 = pd.DataFrame(index=df.index)

# Energy readings: heat pump, hot water, fan and other uses
df2['e_hp'] = df[['HP_in_Tot', 'HP_out_Tot']].sum(axis=1)
df2['e_dhw'] = df['HW_Tot']
df2['e_fan'] = df['Fantech_Tot']
df2['e_other'] = df['main_Tot'] - df2['e_hp'] - df2['e_dhw'] - df2['e_fan']

# Temperatures: indoor, garage, ventilation supply, and outdoor
df2['ti'] = (df[['Din_tmp_Avg', 'Grt_tmp_Avg', 'Brkf_tmp_Avg', 'Kit_tmp_Avg',
                 'BedM_tmp_Avg', 'Bed3_tmp_Avg', 'Bed2_tmp_Avg', 'BedB_tmp_Avg',
                 'Mbath_tmp_Avg']].mean(axis=1) - 32 ) * 5/9
df2['tg'] = (df['garage_tmp_Avg'] - 32) * 5/9
df2['ts'] = (df['FanTsup_RH_Avg'] - 32) * 5/9
df2['te'] = (df['Outside_Tmp_Avg'] - 32) * 5/9

# Other weather variables: solar irradiance and wind speed
df2['i_sol'] = df['SlrW1_Avg']
df2['wind_speed'] = df['wind_speed_mean']

# Let's only keep winter for now
df2.drop(df2.index[(df2.index < pd.to_datetime('2013-11-01')) |
                    (df2.index >= pd.to_datetime('2014-04-01'))], inplace=True)
```

The following block creates a bokeh plot of the variables we just selected. This library offers convenient features such as the ability to zoom in or pan on a graph, and to hide legend entries by clicking them.


```python
palette = Category10[5]

p1 = figure(x_axis_type="datetime", y_range=(-20, 40), plot_width=800, plot_height=300)
p1.line(df2.index, df2['ti'], color=palette[0], legend='Indoor temperature')
p1.line(df2.index, df2['tg'], color=palette[1], legend='Garage temperature')
p1.line(df2.index, df2['te'], color=palette[2], legend='Outdoor temperature')
p1.line(df2.index, df2['ts'], color=palette[3], legend='Ventilation supply temperature')
p1.legend.location = "bottom_right"
p1.legend.click_policy="hide"

p2 = figure(x_axis_type="datetime", x_range=p1.x_range, plot_width=800, plot_height=300)
p2.line(df2.index, df2['e_hp'], color=palette[0], legend='e_hp')
p2.line(df2.index, df2['e_dhw'], color=palette[1], legend='e_dhw')
p2.line(df2.index, df2['e_fan'], color=palette[2], legend='e_fan')
p2.line(df2.index, df2['e_other'], color=palette[3], legend='e_other')
p2.line(df2.index, df2['i_sol'], color=palette[4], legend='i_sol')
p2.legend.location = "top_right"
p2.legend.click_policy="hide"

p3 = figure(x_axis_type="datetime", x_range=p1.x_range, plot_width=800, plot_height=300)
p3.line(df2.index, df2['wind_speed'], color=palette[0], legend='wind_speed')
p3.legend.location = "top_right"

show(column(p1, p2, p3))
```

{% include epa_linreg_bokeh.html %}

## The model

Since we have time series data, the most informative way to use it would be a time series model, typically of the family of autoregressive models with exogenous variables, or an RC state-space model. This would allow us to identify the influences on the dynamics of the output variable.

The present notebook however proposes a more simple, stationary balance equation. This is the full model equation that we are going to consider, supposing that the heat pump is operating in winter conditions:

$$ \Phi_{hp} + \Phi_s + \Phi_v + \Phi_{inf} = H \, (T_i-T_e) + H_g \, (T_i-T_g) $$

On the left side are the heat sources $\Phi$ (W), some of which may be negative:
* $$\Phi_{hp} \propto e_{hp}$$ is the heating power provided by the heat pump to the indoor space. It is proportional to the energy reading $$e_{hp}$$ (Wh), which we will use as output variable, and to the time step size and the COP of the heat pump, supposed constant.
* $$\Phi_s = A_s\, I_{sol}$$ are the solar gains, supposed proportional to the measured outdoor solar irradiance $$I_{sol}$$ (W/m$$^2$$) and an unknown constant solar aperture coefficient $$A_s$$ (m$$^2$$).
* $$\Phi_v = \dot{m} \, c_p \, (T_s-T_i)$$ is the ventilation heat input, with a ventilation supply rate $$\dot{m}$$ and supply temperature $$T_s$$, which is measured (the house has a mechanical ventilation system with heat recovery)
* $$\Phi_{inf} \sim V_{ws} (T_e-T_i)$$ is the heat input from air infiltration. We suppose it is proportional to the wind speed $$V_{ws}$$ and the outdoor-indoor temperature difference.

On the right side are two terms of heat loss through the envelope:
* $$H \, (T_i-T_e)$$ is the direct heat loss from the heated space at temperature $$T_i$$ and the outdoor at $$T_e$$
* $$H_g \, (T_i-T_g)$$ is the heat loss through the partition wall between the heated space and an unheated garage at $$T_g$$.

Linear regression should allow us to identify the coefficients of each term, supposing that they have enough variability and influence on the output $$\Phi_{hp}$$. The outcome of the regression method will let us judge if this hypothesis is appropriate.

In the next step, we create new features in the `df2` dataset to match the hypothesis of this model. Next, data are resampled over daily steps, in order to allow the hypothesis of stationary conditions.


```python
df2['tits'] = df2['ti'] - df2['ts']
df2['vtite'] = df2['wind_speed'] * (df2['ti'] - df2['te'])
df2['tite'] = df2['ti'] - df2['te']
df2['titg'] = df2['ti'] - df2['tg']

df_day = df2.resample('1D').mean()
```

## Training

### First simple model

Before fitting the full model shown above, let us try one with a single explanatory variable, which we assume has the most influence on the energy use of the heat pump: the heat transmission through the envelope.

$$ e_{hp} = a_1 (T_i-T_e)$$

where the $a_1$ parameter includes the heat loss coefficient $$H$$, the COP of the heat pump and the time step size. Since the COP is unknown, we won't be able to estimate $H$. This is fine, as the point of the exercise is mostly to identify influential features.


```python
# Choosing output and inputs
y = df_day['e_hp']
x = df_day[['tite']]

# Model fitting
res = sm.OLS(endog=y, exog=x).fit()

# Summary of the results in a table
print(res.summary())

# Scatter plot of the fitted model
fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(res, 0, ax=ax)
ax.set_ylabel('Heat pump energy (Wh)')
ax.set_xlabel('$T_i - T_e$ (C)')
plt.show()
```

                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                   e_hp   R-squared:                       0.817
    Model:                            OLS   Adj. R-squared:                  0.816
    Method:                 Least Squares   F-statistic:                     670.2
    Date:                Wed, 01 Apr 2020   Prob (F-statistic):           3.30e-57
    Time:                        13:52:50   Log-Likelihood:                -923.13
    No. Observations:                 151   AIC:                             1848.
    Df Residuals:                     150   BIC:                             1851.
    Df Model:                           1
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    tite          13.2377      0.511     25.888      0.000      12.227      14.248
    ==============================================================================
    Omnibus:                       64.945   Durbin-Watson:                   0.389
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              160.288
    Skew:                           1.857   Prob(JB):                     1.56e-35
    Kurtosis:                       6.417   Cond. No.                         1.00
    ==============================================================================

    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


<img src="images/data/epa_linreg_1.png" style="width: 400px;">

The table displays the results of the linear regression fitting by ordinary least squares. Some indicators are useful to judge if the model sufficiently explains the output data, or if some input features are redundant.
* The t-statistic and p-value indicate whether an input has a significant influence on the input: `P>|t|` should be close to zero, meaning that the null hypothesis should be rejected. In this case, the only input is relevant.
* R-squared measures the goodness of fit of the regression. 0.817 is a rather low value, which hints that the output should be explained by additional features in the model.
* AIC and BIC will be used to compare several models. A lower value is preferred.
* A low Durbin-Watson statistic suggests a high autocorrelation of residuals, which means that the model structure is inappropriate.

The plot confirms that the data is not solely explained by a linear function of $$(T_i-T_e)$$, and the model should be improved.

### Complete model

Now we can try a more complete linear regression model, which matches the full model described earlier

$$ e_{hp} = a_1 (T_i-T_e) + a_2 (T_i-T_g) + a_3 I_{sol} + a_4 (T_i-T_s) + a_4 V_{ws}(T_i-T_e) $$

This model has five inputs, which we defined as functions of the columns of the original dataset.


```python
# Model definition and fitting
y = df_day['e_hp']
x = df_day[['tite', 'titg', 'i_sol', 'tits', 'vtite']]
res = sm.OLS(endog=y, exog=x).fit()
print(res.summary())
```

                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                   e_hp   R-squared:                       0.866
    Model:                            OLS   Adj. R-squared:                  0.861
    Method:                 Least Squares   F-statistic:                     188.1
    Date:                Wed, 01 Apr 2020   Prob (F-statistic):           8.93e-62
    Time:                        13:52:50   Log-Likelihood:                -899.85
    No. Observations:                 151   AIC:                             1810.
    Df Residuals:                     146   BIC:                             1825.
    Df Model:                           5
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    tite          22.3649      2.203     10.151      0.000      18.010      26.719
    titg           1.2207      5.845      0.209      0.835     -10.332      12.773
    i_sol         -0.3868      0.202     -1.917      0.057      -0.785       0.012
    tits          -8.2306      3.605     -2.283      0.024     -15.354      -1.107
    vtite         -0.1931      0.710     -0.272      0.786      -1.596       1.209
    ==============================================================================
    Omnibus:                       28.379   Durbin-Watson:                   0.585
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               38.606
    Skew:                           1.066   Prob(JB):                     4.14e-09
    Kurtosis:                       4.263   Cond. No.                         97.1
    ==============================================================================

    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


The R-squared has improved, and the AIC and BIC criteria have decreased: this model seems to be a better choice than the first one.

Two input variables however have a very high $p$-value: $$(T_i-T_g)$$ and $$V_{ws}(T_i-T_e)$$. This suggests that the heat transfer between the heated space and the garage, and the wind, have little impact on the energy consumption of the heat pump. We can simplify the model by removing these two features:


```python
y = df_day['e_hp']
x = df_day[['tite', 'i_sol', 'tits']]
res = sm.OLS(endog=y, exog=x).fit()
print(res.summary())
```

                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                   e_hp   R-squared:                       0.866
    Model:                            OLS   Adj. R-squared:                  0.863
    Method:                 Least Squares   F-statistic:                     317.5
    Date:                Wed, 01 Apr 2020   Prob (F-statistic):           3.04e-64
    Time:                        13:52:50   Log-Likelihood:                -899.93
    No. Observations:                 151   AIC:                             1806.
    Df Residuals:                     148   BIC:                             1815.
    Df Model:                           3
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    tite          22.4237      1.783     12.573      0.000      18.899      25.948
    i_sol         -0.3878      0.189     -2.050      0.042      -0.762      -0.014
    tits          -7.7897      2.601     -2.995      0.003     -12.930      -2.650
    ==============================================================================
    Omnibus:                       27.272   Durbin-Watson:                   0.591
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               36.240
    Skew:                           1.048   Prob(JB):                     1.35e-08
    Kurtosis:                       4.170   Cond. No.                         46.0
    ==============================================================================

    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

With less dimensions, the AIC and BIC criteria have decreased. Furthermore, the R-squared was not impacted by the removal of two features, suggesting that they were indeed not influential.

This model seems to be a decent compromise, although some influences still appear to be missing.

### Adding features

The first equation shown above includes the influences that we assumed the most relevant to the energy consumption of the heat pump. The data may contain some additional explanatory variables, which may help predict $$e_{hp}$$ outside of this formalisation. For instance, the energy for DHW production $$e_{dhw}$$ and other uses $$e_{other}$$ indicate occupancy, which could be correlated to $$e_{hp}$$.


```python
y = df_day['e_hp']
x = df_day[['tite', 'titg', 'tits', 'e_dhw', 'e_other']]
res = sm.OLS(endog=y, exog=x).fit()
print(res.summary())

# Scatter plot of the fitted model
fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(res, 0, ax=ax)
ax.set_ylabel('Heat pump energy (Wh)')
ax.set_xlabel('$T_i - T_e$ (C)')
plt.show()
```

                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                   e_hp   R-squared:                       0.909
    Model:                            OLS   Adj. R-squared:                  0.906
    Method:                 Least Squares   F-statistic:                     292.1
    Date:                Wed, 01 Apr 2020   Prob (F-statistic):           3.87e-74
    Time:                        13:52:50   Log-Likelihood:                -870.33
    No. Observations:                 151   AIC:                             1751.
    Df Residuals:                     146   BIC:                             1766.
    Df Model:                           5
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    tite          23.4624      1.657     14.157      0.000      20.187      26.738
    titg          19.3457      4.825      4.010      0.000       9.810      28.881
    tits         -10.3815      2.062     -5.035      0.000     -14.457      -6.306
    e_dhw         -2.0580      0.429     -4.796      0.000      -2.906      -1.210
    e_other       -0.3216      0.083     -3.890      0.000      -0.485      -0.158
    ==============================================================================
    Omnibus:                       10.411   Durbin-Watson:                   0.832
    Prob(Omnibus):                  0.005   Jarque-Bera (JB):               10.964
    Skew:                           0.659   Prob(JB):                      0.00416
    Kurtosis:                       3.085   Cond. No.                         203.
    ==============================================================================

    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


<img src="images/data/epa_linreg_2.png" style="width: 400px;">

Using the same indicators as before, it seems that the model has been improved again.

As a further improvement, we could suggest using qualitative features such as the day of the week. The energy signature notebook shows that in some buildings, this information is very relevant on the energy use. This building is however a house, with probably less difference between week days and week ends than in office buildings.
