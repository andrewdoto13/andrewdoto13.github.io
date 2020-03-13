---
title: "Renewal Rate Distribution Analysis"
classes: wide
---


# Analyzing renewal rates using distributions

### Background

One of the things we do in the division I work in is to try to project what the cost will be for the various grants and scholarships that are awarded through the office for the following year. This is important because these scholarships are not "funded"; there is no investment account earning interest or any donor supplying the funds. The money used to award these grants/scholarships comes from tuition dollars. So really, you want to try to plan accordingly to budget what is necessary to supply the money for these awards. So in order to do this, or at least one of the elements of this, is that you have to say, "OK, how many freshman students that have the award **right now**, will meet the criteria to renew for their sophomore year?" And then "How many sophomore students that have the award **right now**, will meet the criteria to renew for their junior year", and so on. The rates at which the students meet the criteria is what we call the renewal rate. For example, let's say for the aid year 2017-2018, 1000 freshmen students were given the award. And then at the end of that aid year (which is basically the same as an academic year), 500 of those freshmen met the criteria. In that case, the renewal rate for that award in 2017-2018 is 50%.

### The award in question

For this project, I'll just focus on the most prominent and highly regarded award, the Golden Grizzly Guarantee (formerly the 100% Tuition Guarantee). This award is a need-based award, and in terms of its role in enrollment management strategy, it is perhaps one of the most aggressive awards amongst state universities in Michigan. For this award, if an undergraduate student has an Estimated Family Contribution (EFC) of less than 8000 dollars, this award will cover the full tuition for up to four years. Now, this award is what we call "last dollar"; this means that it takes the cost of tuition, minus the EFC, and then minus any other gift aid (grants, scholarships, etc), and then anything left over is covered by that award. In order to renew the award for the following year, the student has to reach a 2.0 GPA and earn 28 credits. This requirement is to encourage students to stay on track to graduate in 4 years.

### The current method

For planning and budgeting purposes, what we do now is that we take the renewal rate for the prior year and use that rate for the following year. So for example, let's say that in 2017-2018, 50% of freshman met the renewal criteria. So if we are now in 2018-2019, we are trying to plan for 2019-2020. We need to project how many of the freshman students that got the award in 2018-2019 will renew. Well, we know (at this point in time) that the 2017-2018 freshman had a renewal rate of 50%. So, we use that rate to include in our projections for how many sophomores will get the award in 2019-2020.


### Let's try something different: a random walk

After taking a stats course and learning more about the applications of probability distributions, I thought that we could take advantage of the distribution of renewal rates for each of the student classes. We can select a random rate from that distribution to then apply it to the budget; this concept is called a random walk. I think I'd like to provide some sources or videos that go through some of the stats/data science concepts in this article, and perhaps I'll add that down the line. But for now, YouTube, edX, DataCamp, google, StackOverflow, etc. are seriously amazing to find out basically anything nowadays, if you find yourself interested in learning more about a particular thing.

Basically, my thought process is that we can take the ten year historical data for the Golden Grizzly Guarantee (GGG) and find the mean and standard deviation of renewal rates for each class (Freshman, Sophomore, etc). We can then use python's scipy module to construct a normal distribution using the mean and standard deviation as the parameters, from which we can take random samples or selections from to model the future. I remember using a random walk to predict future stock prices in a DataCamp course recently, so this is sort of a similar process.

For this project, I've used python as the tool of choice and the development environment was a Jupyter Notebook. The code displayed in the sections below are actually what I used; the only change would be where I hid the pidms (database index) when I show an example rowset of the data I pulled from the database. As I go along showing you the code snippets, I'll describe my thought process and what the code is doing.

Let's begin by retrieving some data from the database!

The code snippet below shows the query that I used to get the data. The SQL displayed here wasn't run through the Jupyter Notebook I used for this project, but I have done it previously where you can connect to the database straight from the Jupyter Notebook (maybe a future post??). It's pretty sweet and useful, actually! In this case, I executed the query in Oracle SQL Developer and then saved the export to an xlsx file.


```python
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

I'm going to use just a very basic complement of libraries here, all standard for simple data sciency stuff.


```python
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')
from scipy.stats import norm
```

Next I read in the data from the xlsx file I saved and then changed the column names to lowercase using a list comprehension. I'll go ahead and make a copy of the dataframe and hide the pidms, which are just a unique number assigned to a person in the database we have here on campus. I'll hide it just for safe measure, and then show you the first few records of what the data looks like.


```python
df = pd.read_excel("GGG renewals.xlsx")
df.columns = [col.lower() for col in df.columns]
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



So there's a column for the pidm, the aid year that the award pertains to, and the class of the student for that aid year. The renew column is the result of the subquery I used, which looks into the aid year **after** the aid year for that row, and checks to see if they had the award.

Next, I create an object with a list of student classes to validate we have what we are looking for.


```python
classes = df.clas_desc.unique().tolist()
print(classes)
```

    ['Freshman', 'Sophomore', 'Junior', 'Senior']


Now I'll go ahead and create a separate object for each class. Basically I've filtered the main dataset by the student class and then put them into their own object


```python
fr = df.loc[df.clas_desc == "Freshman"]
so = df.loc[df.clas_desc == "Sophomore"]
ju = df.loc[df.clas_desc == "Junior"]
se = df.loc[df.clas_desc == "Senior"]
```

Now for each of the objects, we'll create pivot table that provides the count of students grouped by aid year and by their renewal status. You'll see an example for the freshmen below.


```python
fr_counts = fr.pivot_table(index = 'aidy', columns='renew', values='pidm', aggfunc='count')
so_counts = so.pivot_table(index = 'aidy', columns='renew', values='pidm', aggfunc='count')
ju_counts = ju.pivot_table(index = 'aidy', columns='renew', values='pidm', aggfunc='count')

print(fr_counts)
```

    renew  Not renewed  Renewed
    aidy                       
    910          300.0    178.0
    1011         302.0    177.0
    1112         449.0    167.0
    1213         193.0    101.0
    1314         187.0    136.0
    1415         187.0    134.0
    1516         310.0    174.0
    1617         261.0    196.0
    1718         317.0    155.0
    1819         372.0    231.0
    1920         616.0      NaN


So what we have now are objects that have each aid year as a row, and then a count of the students that renewed or not as columns. A **really** nice feature about the pandas library is that you can use pandas objects of similar length and perform operations on them very easily and efficiently. So very simply, for each object, we will take the count of renewed students and then divide it by the total students. In a very nice and easy way thanks to pandas, we get an object with this operation for each aid year. So in effect, we've just calculated the renewal rates for each aid year.


```python
fr_rates = fr_counts.Renewed / (fr_counts.Renewed + fr_counts['Not renewed'])
so_rates = so_counts.Renewed / (so_counts.Renewed + so_counts['Not renewed'])
ju_rates = ju_counts.Renewed / (ju_counts.Renewed + ju_counts['Not renewed'])

print(fr_rates)
```

    aidy
    910     0.372385
    1011    0.369520
    1112    0.271104
    1213    0.343537
    1314    0.421053
    1415    0.417445
    1516    0.359504
    1617    0.428884
    1718    0.328390
    1819    0.383085
    1920         NaN
    dtype: float64


Now that we have the renewal rates, let's put them together into one pandas dataframe. Note that we've disregarded seniors, as it's not relevant to our project here. Using the pandas concat function, I've sorta "stacked" the three columns horizontally. So what we see as a result is a dataframe with a row for each aid year and then a column for each student class. You also may notice that I also selected out the final row, and that's because we don't have the data for the following year yet!


```python
all_rates = pd.concat([fr_rates,so_rates, ju_rates], axis=1)
all_rates.columns = classes[0:3]
all_rates = all_rates.iloc[0:-1]

print(all_rates)
```

          Freshman  Sophomore    Junior
    aidy                               
    910   0.372385        NaN       NaN
    1011  0.369520   0.457143       NaN
    1112  0.271104   0.470930  0.556962
    1213  0.343537   0.544910  0.545455
    1314  0.421053   0.564356  0.552381
    1415  0.417445   0.534351  0.529412
    1516  0.359504   0.601449  0.640000
    1617  0.428884   0.569061  0.684783
    1718  0.328390   0.524510  0.715447
    1819  0.383085   0.574586  0.620968


OK, so now let's take a look at the distribution of the renewal rates and check them out over time. Matplotlib's pyplot and seaborn make it very easy to make charts and plots, so that's what you'll see below.


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



<img src="{{ site.url }}{{ site.baseurl }}/images/renewalrate/fig1.png" alt="">



```python
all_rates.plot(kind = 'bar', figsize = (15,5))
plt.ylabel("Renewal Rate")
plt.xlabel("Aid Year")
plt.title("Class Renewal Rates by Aid Year")
```




<img src="{{ site.url }}{{ site.baseurl }}/images/renewalrate/fig2.png" alt="">


OK, so from the very beginning, you'll notice how I imported "norm" from the scipy package. This is the part of the scipy package that contains the functions and classes related to the normal distribution. I didn't know beforehand for sure what kind of distribution the renewal rates were going to look like, but I guess I was just hoping that they were at least close to normal. If you look at the distribution plot (the first figure),


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
