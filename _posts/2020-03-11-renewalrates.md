```python
#Here is the query used to collect the data:

query = '''
select
    a.rprawrd_pidm pidm
    , a.rprawrd_aidy_code aidy
    , b.rfrbase_fund_title title
    , case when
            (
                select
                    innera.rprawrd_fund_code
                from
                    faismgr.rprawrd innera
                where
                    innera.rprawrd_pidm = a.rprawrd_pidm
                    and
                    innera.rprawrd_aidy_code = a.rprawrd_aidy_code + 101
                    and
                    innera.rprawrd_fund_code = 'F317'
                    and
                    innera.rprawrd_paid_amt > 0
            ) = 'F317'
    then 'Renewed'
    else 'Not renewed'
    end as renew
    , s.clas_desc
from
    faismgr.rprawrd a
    inner join
    faismgr.rfrbase b
    on
    a.rprawrd_fund_code = b.rfrbase_fund_code
    inner join
    baninst1.as_student_enrollment_summary s
    on
    a.rprawrd_pidm = s.pidm_key
    and
    concat(concat(20,substr(a.rprawrd_aidy_code,1, 2)), 40) = s.term_code_key
where
    a.rprawrd_fund_code in ('F301','F317')
    and
    a.rprawrd_paid_amt > 0
order by 1,2
'''

```

# Analyzing renewal rates using distributions

### Background

One of the things we do in the division I work in is to try to project what the cost will be for the various grants and scholarships that are awarded through the office for the following year. This is important because these scholarships are not "funded"; there is no investment account earning interest or any donor supplying the funds. The money used to award these grants/scholarships comes from tuition dollars. So really, you want to try to plan accordingly to budget what is necessary to supply the money for these awards for the students that get them. So in order to do this, or at least one of the elements of this, is that you have to say, "OK, how many freshman students that have the award **right now**, will meet the criteria to renew for their sophomore year?" And then "How many sophomore students that have the award **right now**, will meet the criteria to renew for their junior year", and so on. The rates at which the students meet the criteria is what we call the renewal rate. For example, let's say in for the aid year 2017-2018, 1000 freshmen students were given the award. And then at the end of that aid year(which is basically the same as an academic year), 500 of those freshmen met the criteria. In that case, the renewal rate for that award in 2017-2018 is 50%.

### The award in question

For this project, I'll just focus on the most prominent and highly regarded award, the Golden Grizzly Guarantee(formerly the 100% Tuition Guarantee). This award is a need-based award, and in terms of its role in enrollment management strategy, it is perhaps one of the most aggressive awards amongst state universities in Michigan. For this award, if an undergraduate student has an Estimated Family Contribution of less than 8000 dollars, this award will cover the full tuition for up to four years. Now, this award is what we call "last dollar"; this means that it takes the cost of tuition minus the EFC, and then minus any other gift aid(grants, scholarships, etc), and then anything left over is covered by that award.

### The current method

For planning and budgeting purposes, what we do now is that we take the renewal rate for the prior year and use that rate for the following year. So for example, let's say that in 2017-2018, 50% of freshman met the renewal criteria. So if we are now in 2018-2019, we are trying to plan for 2019-2020. We need to project how many of the freshman students that got the award in 2018-2019 will renew. Well, we know (at this point in time) know that the 2017-2018 freshman had a renewal rate of 50%. So, we use that rate to include in our projections for how many sophomores will get the award in 2019-2020. The timeframes can be kinda confusing, I sure as heck get confused! Hopefully this helps:

  2017-2018                   ------------------------------------->      2018-2019          --------------------------------------->    2019-2020 (budget)

50% Renewal Rate                                                  1200 Freshmen got award                                    50% renewal rate x 1200 freshmen = 600 sophomores


### Let's try something different: random walk

After taking a stats course and learning more about the applications of probability distributions, I thought that we could take advantage of the distribution of renewal rates for each of the student classes, and selecting a random rate from that distribution to then apply it to the budget. This concept is called a random walk. I think I'd like to provide some sources or videos that go through some of the concepts in this article, and perhaps I'll add that down the line. But for now, YouTube, edX, DataCamp, google, StackOverflow, etc. are seriously amazing to find out basically anything nowadays. Basically, my thought process is that we can take the ten year historical data for the Golden Grizzly Guarantee (GGG) and find the mean and standard deviation of renewal rates for each class (Freshman, Sophomore, etc). We then use python's scipy module to construct a normal distribution using the mean and standard deviation as the parameters, from which we can take random samples or selections from to model the future. I remember using a random walk to predict future stock prices in a DataCamp course recently, so this is sort of a similar process.


```python
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')
from scipy.stats import norm
```


```python
df = pd.read_excel("GGG renewals.xlsx")
```


```python
df.columns = [col.lower() for col in df.columns]
```


```python
classes = df.clas_desc.unique().tolist()
```


```python
fr = df.loc[df.clas_desc == "Freshman"]
so = df.loc[df.clas_desc == "Sophomore"]
ju = df.loc[df.clas_desc == "Junior"]
se = df.loc[df.clas_desc == "Senior"]
```


```python
fr_counts = fr.pivot_table(index = 'aidy', columns='renew', values='pidm', aggfunc='count')
so_counts = so.pivot_table(index = 'aidy', columns='renew', values='pidm', aggfunc='count')
ju_counts = ju.pivot_table(index = 'aidy', columns='renew', values='pidm', aggfunc='count')
se_counts = se.pivot_table(index = 'aidy', columns='renew', values='pidm', aggfunc='count')
```


```python
fr_rates = fr_counts.Renewed / (fr_counts.Renewed + fr_counts['Not renewed'])
so_rates = so_counts.Renewed / (so_counts.Renewed + so_counts['Not renewed'])
ju_rates = ju_counts.Renewed / (ju_counts.Renewed + ju_counts['Not renewed'])
se_rates = se_counts.Renewed / (se_counts.Renewed + se_counts['Not renewed'])
```


```python
all_rates = pd.concat([fr_rates,so_rates, ju_rates], axis=1)
all_rates.columns = classes[0:3]
all_rates = all_rates.iloc[0:-1]

```


```python
plt.figure(figsize=(15,5))
sns.distplot(all_rates.Freshman, color='green', label='Freshman', norm_hist=True, kde=True)
sns.distplot(all_rates.Sophomore.dropna(), color='red', label="Sophomore", norm_hist=True)
sns.distplot(all_rates.Junior.dropna(), color='blue', label="Junior", norm_hist=True)
plt.legend(loc = 'upper left')
plt.xlabel("Renewal Rate")
plt.ylabel("Frequency")
plt.title("Distribution of Renewal Rates by Class")
```




    Text(0.5, 1.0, 'Distribution of Renewal Rates by Class')




![png](output_11_1.png)



```python
all_rates.plot(kind = 'bar', figsize = (15,5))
plt.ylabel("Renewal Rate")
plt.xlabel("Aid Year")
plt.title("Class Renewal Rates by Aid Year")
```




    Text(0.5, 1.0, 'Class Renewal Rates by Aid Year')




![png](output_12_1.png)



```python
fr_mean, fr_std = all_rates.Freshman.mean(), all_rates.Freshman.std()
so_mean, so_std = all_rates.Sophomore.mean(), all_rates.Sophomore.std()
ju_mean, ju_std = all_rates.Junior.mean(), all_rates.Junior.std()
```


```python
fr_dist = norm(loc = fr_mean, scale = fr_std).rvs(size = 250)
so_dist = norm(loc = so_mean, scale = so_std).rvs(size = 250)
ju_dist = norm(loc = ju_mean, scale = ju_std).rvs(size = 250)
```


```python
fr_conf_int = (round(fr_mean - norm.ppf(0.975) * fr_std,3), round(fr_mean + norm.ppf(0.975) * fr_std, 3) )
so_conf_int = (round(so_mean - norm.ppf(0.975) * so_std, 3), round(so_mean + norm.ppf(0.975) * so_std, 3) )
ju_conf_int = (round(ju_mean - norm.ppf(0.975) * ju_std,3), round(ju_mean + norm.ppf(0.975) * ju_std, 3))
```


```python
print("The 95% confidence interval for Freshman renewal rate is: ", fr_conf_int)
print("\nThe 95% confidence interval for Sophomore renewal rate is: ", so_conf_int)
print("\nThe 95% confidence interval for Junior renewal rate is: ", ju_conf_int)
```

    The 95% confidence interval for Freshman renewal rate is:  (0.275, 0.464)

    The 95% confidence interval for Sophomore renewal rate is:  (0.444, 0.632)

    The 95% confidence interval for Junior renewal rate is:  (0.468, 0.743)



```python
df_show = df.copy()
df_show['pidm'] = "Hidden"
df_show.head()
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
      <th></th>
      <th>pidm</th>
      <th>aidy</th>
      <th>title</th>
      <th>renew</th>
      <th>clas_desc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hidden</td>
      <td>1011</td>
      <td>100% Tuition Grant</td>
      <td>Not renewed</td>
      <td>Freshman</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hidden</td>
      <td>910</td>
      <td>100% Tuition Grant</td>
      <td>Not renewed</td>
      <td>Freshman</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hidden</td>
      <td>910</td>
      <td>100% Tuition Grant</td>
      <td>Renewed</td>
      <td>Freshman</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hidden</td>
      <td>1011</td>
      <td>100% Tuition Grant Renewal</td>
      <td>Not renewed</td>
      <td>Sophomore</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hidden</td>
      <td>910</td>
      <td>100% Tuition Grant</td>
      <td>Not renewed</td>
      <td>Freshman</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
