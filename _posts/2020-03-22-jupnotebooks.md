---
title: "Using Jupyter Notebooks for transparency and reproducibility"
classes: wide
tags: [python, jupyter notebooks]
desc: "Jupyter Notebooks are a great tool for everyday analysis and is my go-to tool"
---

# Using Jupyter Notebooks for transparency and reproducibility

A Jupyter Notebook is an amazing tool. I was introduced to Jupyter Notebooks in the beginning of my coursework through IU's data science program, and aside from python and its data science libraries, it has been probably the most transformational thing/tool I've learned since I've gotten interested in analytics. You can sing the praises of Jupyter Notebooks until the cows come home, but for this post, I'd like to focus on how it helps with transparency and reproducibility.

Often times at my past few jobs as well as currently, I'll get a descriptive question that I would have to answer, like "Which of the classes we offered this past academic year had the highest and lowest average GPA?" or "Of the non-resident students that got XYZ award this past year, what other awards did they get and how much in total did they get?". Now, these aren't necessarily the toughest questions in the world to answer, but they can get fairly complicated. But, when you have to revisit that question months later, it gets difficult to remember your exact approach to the problem. In the past, I would pull data from the database into an excel sheet and do whatever filtering/pivoting/vlookup-ing to find the answer that way. But when you have to revisit that excel sheet months later, unless you had really good supplemental documentation, it can be a challenge.

This is where I've really found Jupyter Notebooks to be extremely useful; it has made me that much more effective and more confident in my work. The way it's constructed by splitting the code into cell blocks and then how the execution is interactive at each block is just immense. It's also helped me in my learning by helping me develop my problem-solving skills by splitting up problems into component parts. Jupyter Notebooks are perfect for that, and the fact that they are free to use is plain awesome. When I try to answer descriptive questions or do really any kind of analysis, I go straight for a Jupyter Notebook. When I have to share the information in a meeting or answer questions about my process, I can simply share the notebook or pull it up, so I can share the exact steps I took to meet the objective. Not only that, but because the output from any individual cell can be displayed, you can sort of check your assumptions about what a function does or whatever code you wrote is *actually* doing. It has made me a better analyst for sure, so I feel obligated to share my experience in case it can help anyone else in their work or learning.

Recently, one of the projects I had to do was to analyze one of the awards we have called the Detroit Promise. Here is the link in case you are interested in learning more about this award: https://www.detroitchamber.com/econdev/education-and-talent/detroit-promise/.

What I needed to figure out was the difference between what the students were awarded for their Detroit Promise award vs. what they would have gotten in a hypothetical Golden Grizzly Guarantee (GGG, formerly 100% Tuition Guarantee). To do this, I had to pull the concurrent awards each student had and then how much that award was for.

The following parts of this post will be the Jupyter Notebook cells that I used to complete this analysis. The first code block below is the SQL query that I wrote to pull the data from the database. I took a slightly different approach to pulling the data this time. I used a few common table expressions (CTE) and then left joined them together. Essentially what this gives me is a unique row for every award that a Detroit Promise student got. But, the first few columns of the dataset contain the Detroit Promise award details, so that a row for that award is not needed. I did it this way because I thought that it would make cleaning the data a little bit easier, but there are many ways you could do this, of course. If you aren't familiar with SQL, skip this next block and continue on!


```python
query = '''
with
    d as
        (
            select
                a.rprawrd_pidm pidm
                , a.rprawrd_aidy_code aidy
                , b.rfrbase_fund_title title
                , a.rprawrd_awst_code awst
                , a.rprawrd_offer_amt offer
                , a.rprawrd_paid_amt paid
            from
                faismgr.rprawrd a
                inner join
                faismgr.rfrbase b
                on
                a.rprawrd_fund_code = b.rfrbase_fund_code
            where
                a.rprawrd_fund_code = 'F430'
        ),
    a as
        (
            select
                a.rprawrd_pidm pidm
                , a.rprawrd_aidy_code aidy
                , b.rfrbase_fund_title title
                , b.rfrbase_fsrc_code fsrc
                , b.rfrbase_ftyp_code ftyp
                , a.rprawrd_awst_code awst
                , a.rprawrd_offer_amt offer
                , a.rprawrd_paid_amt paid
            from
                faismgr.rprawrd a
                inner join
                faismgr.rfrbase b
                on
                a.rprawrd_fund_code = b.rfrbase_fund_code
            where
                a.rprawrd_aidy_code >= '1617'
        ),
    r as
        (
            select
                v.rnvand0_pidm pidm
                , v.rnvand0_aidy_code aidy
                , v.rnvand0_efc_amt efc
            from
                baninst1.rnvand0 v
            where
                v.rnvand0_aidy_code >= '1617'
        )

select
    d.pidm pidm
    , d.aidy aidy
    , d.title det
    , d.awst det_awst
    , d.offer det_offer
    , d.paid det_paid
    , a.title oth_aw
    , a.fsrc oth_fsrc
    , a.ftyp oth_ftyp
    , a.awst oth_awst
    , a.offer oth_offer
    , a.paid oth_paid
    , r.efc efc

from
    d
    left join
    a
    on
    d.pidm = a.pidm
    and
    d.aidy = a.aidy
    left join
    r
    on
    d.pidm = r.pidm
    and
    d.aidy = r.aidy
where
    a.title <> 'OU Detroit Scholarship'
    '''
```

The Notebook begins by importing some standard libraries. I think I might go through and make a post about the libraries that I like and what I use them for in another post, but these are all pretty basic and standard data science libraries that you'll find that most people use.


```python
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
```

This next block is where I will go ahead and import the data that I saved in an excel file after executing that SQL query to pull it from the database. From now on, since I need to hide the "PIDM" column, I will make a copy of the object that I am using for the data, and then change that column to hide it. I will do that to the data that I print out for you to see, but the dataset looks the same otherwise.


```python
df = pd.read_excel("detprom.xlsx")
df.columns = [col.lower() for col in df.columns]
df_copy = df.copy()
df_copy['pidm'] = 'Hidden'
df_copy.head()
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
      <th>det</th>
      <th>det_awst</th>
      <th>det_offer</th>
      <th>det_paid</th>
      <th>oth_aw</th>
      <th>oth_fsrc</th>
      <th>oth_ftyp</th>
      <th>oth_awst</th>
      <th>oth_offer</th>
      <th>oth_paid</th>
      <th>efc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hidden</td>
      <td>1617</td>
      <td>OU Detroit Scholarship</td>
      <td>CNCL</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>OU Trustee Academic Endow</td>
      <td>ENDW</td>
      <td>SCHL</td>
      <td>ACPT</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>6694</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hidden</td>
      <td>1819</td>
      <td>OU Detroit Scholarship</td>
      <td>CNCL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>OU Trustee Academic Endow</td>
      <td>ENDW</td>
      <td>SCHL</td>
      <td>ACPT</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>15476</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>ACPT</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>OU Trustee Academic Endow</td>
      <td>ENDW</td>
      <td>SCHL</td>
      <td>ACPT</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>6336</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>CNCL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>OU Trustee Academic Endow</td>
      <td>ENDW</td>
      <td>SCHL</td>
      <td>CNCL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>37884</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>ACPT</td>
      <td>472.0</td>
      <td>472.0</td>
      <td>CMI Retention Scholarship</td>
      <td>INST</td>
      <td>SCHL</td>
      <td>ACPT</td>
      <td>500.0</td>
      <td>500.0</td>
      <td>9432</td>
    </tr>
  </tbody>
</table>
</div>



The next thing that I am doing here is that I need to add a column for the tuition for that year. That will be useful for doing calculations for the GGG award. Since I only have three years worth of data, there are three different tuition rates. I suppose I could have written a for loop with conditional flow to do this; that would have been more elegant. But this works, so we'll go with it!


```python
df['tui'] = 12064
df.loc[df.aidy == 1718, 'tui'] = 11970
df.loc[df.aidy == 1819, 'tui'] = 12606
df.loc[df.aidy == 1920, 'tui'] = 13346
df_copy = df.copy()
df_copy['pidm'] = 'Hidden'
df_copy.head()
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
      <th>det</th>
      <th>det_awst</th>
      <th>det_offer</th>
      <th>det_paid</th>
      <th>oth_aw</th>
      <th>oth_fsrc</th>
      <th>oth_ftyp</th>
      <th>oth_awst</th>
      <th>oth_offer</th>
      <th>oth_paid</th>
      <th>efc</th>
      <th>tui</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hidden</td>
      <td>1617</td>
      <td>OU Detroit Scholarship</td>
      <td>CNCL</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>OU Trustee Academic Endow</td>
      <td>ENDW</td>
      <td>SCHL</td>
      <td>ACPT</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>6694</td>
      <td>12064</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hidden</td>
      <td>1819</td>
      <td>OU Detroit Scholarship</td>
      <td>CNCL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>OU Trustee Academic Endow</td>
      <td>ENDW</td>
      <td>SCHL</td>
      <td>ACPT</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>15476</td>
      <td>12606</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>ACPT</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>OU Trustee Academic Endow</td>
      <td>ENDW</td>
      <td>SCHL</td>
      <td>ACPT</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>6336</td>
      <td>13346</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>CNCL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>OU Trustee Academic Endow</td>
      <td>ENDW</td>
      <td>SCHL</td>
      <td>CNCL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>37884</td>
      <td>13346</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>ACPT</td>
      <td>472.0</td>
      <td>472.0</td>
      <td>CMI Retention Scholarship</td>
      <td>INST</td>
      <td>SCHL</td>
      <td>ACPT</td>
      <td>500.0</td>
      <td>500.0</td>
      <td>9432</td>
      <td>13346</td>
    </tr>
  </tbody>
</table>
</div>



So this next code block takes a little bit more explanation but also demonstrates why Jupyter Notebooks are so useful in this type of work. So let me take this step by step. What I need to eventually get to is, for every student and each aid year, the total amount of aid (non-loan, non-work study, non-housing related). I also need the total institutional aid for every student in that aid year (only institutional awards, non-housing related).

So the calc_elig_gift_aid object filters the dataframe for the relevant awards, and then similarly for the inst_aid object. What I will then do is to group the dataframes by the pidm and aid year, summing the amounts for those awards. I can then join these dataframes to the dataframe I already built so far, giving me the columns that I need. Bear with me, this will make sense as I go on.

When I talk about how Jupyter Notebooks help me to be more transparent in my work, this is a perfect example of that. This one step alone is esoteric; I am applying multiple filters two separate times to get the totals for the students. Because I can share this, I can go back and make sure there wasn't anything I missed. That is huge because if there is a case where I have to go back, I can simply change the code and re-run it. It's amazing how easy reproduciblity and transparency are with Jupyter Notebooks.


```python
calc_elig_gift_aid = df.loc[(df.oth_ftyp != 'LOAN') &
                            (df.oth_ftyp != 'WORK') &
                            (df.oth_aw != 'OU Housing Grant') &
                            (df.oth_aw != 'OU Housing Grant Renewal') &
                            (df.oth_aw != 'OU Wayne County Housing Award') &
                            (df.oth_aw != 'OU Wayne County Housing Renew') &
                            (df.oth_aw != '100% Tuition Grant') &
                            (df.oth_aw != 'Housing Award Not Payable') &
                           (df.oth_aw != 'OU Geographic Region Award 11')]

inst_aid = df.loc[(df.oth_fsrc == 'INST') &
                  (df.oth_aw != 'OU Housing Grant') &
                  (df.oth_aw != 'OU Housing Grant Renewal') &
                  (df.oth_aw != 'OU Wayne County Housing Award') &
                  (df.oth_aw != 'OU Wayne County Housing Renew') &
                  (df.oth_aw != '100% Tuition Grant') &
                  (df.oth_aw != 'Housing Award Not Payable') &
                  (df.oth_aw != 'OU Geographic Region Award 11')]

df_copy = inst_aid.copy()
df_copy['pidm'] = 'Hidden'
df_copy.head()
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
      <th>det</th>
      <th>det_awst</th>
      <th>det_offer</th>
      <th>det_paid</th>
      <th>oth_aw</th>
      <th>oth_fsrc</th>
      <th>oth_ftyp</th>
      <th>oth_awst</th>
      <th>oth_offer</th>
      <th>oth_paid</th>
      <th>efc</th>
      <th>tui</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>ACPT</td>
      <td>472.0</td>
      <td>472.0</td>
      <td>CMI Retention Scholarship</td>
      <td>INST</td>
      <td>SCHL</td>
      <td>ACPT</td>
      <td>500.0</td>
      <td>500.0</td>
      <td>9432</td>
      <td>13346</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Hidden</td>
      <td>1819</td>
      <td>OU Detroit Scholarship</td>
      <td>CNCL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>OU Wade McCree Scholarship</td>
      <td>INST</td>
      <td>SCHL</td>
      <td>ACPT</td>
      <td>12892.5</td>
      <td>12892.5</td>
      <td>15476</td>
      <td>12606</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>ACPT</td>
      <td>1667.5</td>
      <td>1667.5</td>
      <td>OU Recognition Award Renewal</td>
      <td>INST</td>
      <td>SCHL</td>
      <td>NORE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6272</td>
      <td>13346</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>ACPT</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>OU Recognition Award Renewal</td>
      <td>INST</td>
      <td>SCHL</td>
      <td>ACPT</td>
      <td>1500.0</td>
      <td>1500.0</td>
      <td>6336</td>
      <td>13346</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>ACPT</td>
      <td>2358.0</td>
      <td>2358.0</td>
      <td>OU First Year Focus Award</td>
      <td>INST</td>
      <td>SCHL</td>
      <td>ACPT</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>0</td>
      <td>13346</td>
    </tr>
  </tbody>
</table>
</div>



Now that we have those 2 new datasets, I am going to use the groupby method to group the dataframe by PIDM and aid year. I will use the sum function as the aggregate method. This will produce a dataframe with three columns, one for the PIDM, the aid year, and then the total amount. I will print out this grouped dataframe for the second part, the institutional aid.


```python
tot_gift = calc_elig_gift_aid.loc[calc_elig_gift_aid.oth_paid > 0].groupby(['pidm','aidy'])['oth_paid'].sum().reset_index()
tot_gift.columns = ['pidm', 'aidy', 'tot_gift_aid']

tot_inst_aid = inst_aid.loc[inst_aid.oth_paid > 0].groupby(['pidm','aidy'])['oth_paid'].sum().reset_index()
tot_inst_aid.columns = ['pidm', 'aidy', 'tot_inst_aid']
df_copy = tot_inst_aid.copy()
df_copy['pidm'] = 'Hidden'
df_copy.head()
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
      <th>tot_inst_aid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>6000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hidden</td>
      <td>1617</td>
      <td>5000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hidden</td>
      <td>1718</td>
      <td>6500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hidden</td>
      <td>1819</td>
      <td>6500.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>6500.0</td>
    </tr>
  </tbody>
</table>
</div>



For this next code block, I will have to filter the original dataframe to get the records where the Detroit Promise award actually paid out, and then I'll select the relevant columns. I'll then drop the duplicates so that the dataframe consists of the unique students that got the Detroit Promise award.


```python
det = df.loc[df.det_paid > 0, ['pidm','aidy','det','det_paid','tui','efc']].drop_duplicates()
df_copy = det.copy()
df_copy['pidm'] = 'Hidden'
df_copy.head()
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
      <th>det</th>
      <th>det_paid</th>
      <th>tui</th>
      <th>efc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>472.00</td>
      <td>13346</td>
      <td>9432</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Hidden</td>
      <td>1718</td>
      <td>OU Detroit Scholarship</td>
      <td>12162.00</td>
      <td>11970</td>
      <td>60760</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hidden</td>
      <td>1819</td>
      <td>OU Detroit Scholarship</td>
      <td>8247.00</td>
      <td>12606</td>
      <td>2195</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>13487.00</td>
      <td>13346</td>
      <td>4923</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Hidden</td>
      <td>1718</td>
      <td>OU Detroit Scholarship</td>
      <td>4301.25</td>
      <td>11970</td>
      <td>2879</td>
    </tr>
  </tbody>
</table>
</div>



Now what I will do combine the two grouped dataframes from above to the det dataframe I just created. I will left join them on the PIDM and the aid year. So once I print them out, you will see that the two columns for the total gift aid and total institutional aid are now added to the dataframe.


```python
df2 = pd.merge(det, tot_gift, how='left', on=['pidm','aidy'])
df3 = pd.merge(df2, tot_inst_aid, how='left', on=['pidm','aidy'])
df_copy = df3.copy()
df_copy['pidm'] = 'Hidden'
df_copy.head()
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
      <th>det</th>
      <th>det_paid</th>
      <th>tui</th>
      <th>efc</th>
      <th>tot_gift_aid</th>
      <th>tot_inst_aid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>472.00</td>
      <td>13346</td>
      <td>9432</td>
      <td>7000.0</td>
      <td>7000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hidden</td>
      <td>1718</td>
      <td>OU Detroit Scholarship</td>
      <td>12162.00</td>
      <td>11970</td>
      <td>60760</td>
      <td>1500.0</td>
      <td>1500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hidden</td>
      <td>1819</td>
      <td>OU Detroit Scholarship</td>
      <td>8247.00</td>
      <td>12606</td>
      <td>2195</td>
      <td>5645.0</td>
      <td>1500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>13487.00</td>
      <td>13346</td>
      <td>4923</td>
      <td>3145.0</td>
      <td>1500.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hidden</td>
      <td>1718</td>
      <td>OU Detroit Scholarship</td>
      <td>4301.25</td>
      <td>11970</td>
      <td>2879</td>
      <td>4770.0</td>
      <td>1500.0</td>
    </tr>
  </tbody>
</table>
</div>



The next thing that I need to do is to create a column for the the hypothetical amount that the student would have gotten for the GGG award. To do this, I will take the tuition and subtract the efc and the total gift aid. This is how the GGG amount is calculated normally, so this will give us an idea as to how much they would have been awarded. Additionally, I used a lambda function to replace any negative amounts with zero to make it easier to work with. After that, I can simply find the difference between the hypothetical GGG and the Detroit Promise award that they got.


```python
df3['hypo_ggg'] = (df3.tui - df3.efc - df3.tot_gift_aid).apply(lambda x: 0 if x < 0 else x)
df3['detprom_ggg_diff'] = df3.det_paid - df3.hypo_ggg
df_copy = df3.copy()
df_copy['pidm'] = 'Hidden'
df_copy.sort_values(['aidy', 'pidm'])
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
      <th>det</th>
      <th>det_paid</th>
      <th>tui</th>
      <th>efc</th>
      <th>tot_gift_aid</th>
      <th>tot_inst_aid</th>
      <th>hypo_ggg</th>
      <th>detprom_ggg_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Hidden</td>
      <td>1718</td>
      <td>OU Detroit Scholarship</td>
      <td>12162.00</td>
      <td>11970</td>
      <td>60760</td>
      <td>1500.0</td>
      <td>1500.0</td>
      <td>0.0</td>
      <td>12162.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hidden</td>
      <td>1718</td>
      <td>OU Detroit Scholarship</td>
      <td>4301.25</td>
      <td>11970</td>
      <td>2879</td>
      <td>4770.0</td>
      <td>1500.0</td>
      <td>4321.0</td>
      <td>-19.75</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Hidden</td>
      <td>1718</td>
      <td>OU Detroit Scholarship</td>
      <td>1242.00</td>
      <td>11970</td>
      <td>36</td>
      <td>12420.0</td>
      <td>6500.0</td>
      <td>0.0</td>
      <td>1242.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hidden</td>
      <td>1819</td>
      <td>OU Detroit Scholarship</td>
      <td>8247.00</td>
      <td>12606</td>
      <td>2195</td>
      <td>5645.0</td>
      <td>1500.0</td>
      <td>4766.0</td>
      <td>3481.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Hidden</td>
      <td>1819</td>
      <td>OU Detroit Scholarship</td>
      <td>3111.50</td>
      <td>12606</td>
      <td>7944</td>
      <td>6500.0</td>
      <td>6500.0</td>
      <td>0.0</td>
      <td>3111.50</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Hidden</td>
      <td>1819</td>
      <td>OU Detroit Scholarship</td>
      <td>3947.00</td>
      <td>12606</td>
      <td>2950</td>
      <td>9945.0</td>
      <td>6500.0</td>
      <td>0.0</td>
      <td>3947.00</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>472.00</td>
      <td>13346</td>
      <td>9432</td>
      <td>7000.0</td>
      <td>7000.0</td>
      <td>0.0</td>
      <td>472.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>13487.00</td>
      <td>13346</td>
      <td>4923</td>
      <td>3145.0</td>
      <td>1500.0</td>
      <td>5278.0</td>
      <td>8209.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>1667.50</td>
      <td>13346</td>
      <td>6272</td>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>2074.0</td>
      <td>-406.50</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>757.50</td>
      <td>13346</td>
      <td>7813</td>
      <td>8000.0</td>
      <td>8000.0</td>
      <td>0.0</td>
      <td>757.50</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>2358.00</td>
      <td>13346</td>
      <td>0</td>
      <td>12195.0</td>
      <td>6000.0</td>
      <td>1151.0</td>
      <td>1207.00</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>12257.50</td>
      <td>13346</td>
      <td>21232</td>
      <td>1500.0</td>
      <td>1500.0</td>
      <td>0.0</td>
      <td>12257.50</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>5229.00</td>
      <td>13346</td>
      <td>5829</td>
      <td>7245.0</td>
      <td>6500.0</td>
      <td>272.0</td>
      <td>4957.00</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>1808.75</td>
      <td>13346</td>
      <td>7248</td>
      <td>6500.0</td>
      <td>6500.0</td>
      <td>0.0</td>
      <td>1808.75</td>
    </tr>
  </tbody>
</table>
</div>



Now that we have the difference for each student, I can simply take the sum of that column; this will tell us how much more the Detroit Promise was, versus the hypothetical GGG.


```python
df3.detprom_ggg_diff.sum()
```




    53186.0



The following code block is useful for inspecting individual students and the awards they got in a particular aid year. What we can do is to pick out a student and then assign the pidm and aid year to those objects and then run the code block. It will print out the information for that student so that we can see what is behind the total figures. I placed the results into an object and then hid the PIDM column. You can see how this can be useful to see what awards and amounts comprised the totals.


```python
stu = # insert pidm
aidy = # insert aid year

ex = calc_elig_gift_aid.loc[(calc_elig_gift_aid.pidm == stu) &
                       (calc_elig_gift_aid.aidy == aidy)].sort_values(by=['aidy','oth_awst'])

ex['pidm'] = "Hidden"

ex
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
      <th>det</th>
      <th>det_awst</th>
      <th>det_offer</th>
      <th>det_paid</th>
      <th>oth_aw</th>
      <th>oth_fsrc</th>
      <th>oth_ftyp</th>
      <th>oth_awst</th>
      <th>oth_offer</th>
      <th>oth_paid</th>
      <th>efc</th>
      <th>tui</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>ACPT</td>
      <td>472.0</td>
      <td>472.0</td>
      <td>CMI Retention Scholarship</td>
      <td>INST</td>
      <td>SCHL</td>
      <td>ACPT</td>
      <td>500.0</td>
      <td>500.0</td>
      <td>9432</td>
      <td>13346</td>
    </tr>
    <tr>
      <th>128</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>ACPT</td>
      <td>472.0</td>
      <td>472.0</td>
      <td>OU Recognition Award 11</td>
      <td>INST</td>
      <td>SCHL</td>
      <td>ACPT</td>
      <td>1500.0</td>
      <td>1500.0</td>
      <td>9432</td>
      <td>13346</td>
    </tr>
    <tr>
      <th>141</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>ACPT</td>
      <td>472.0</td>
      <td>472.0</td>
      <td>OU Golden Grant</td>
      <td>INST</td>
      <td>GRNT</td>
      <td>ACPT</td>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>9432</td>
      <td>13346</td>
    </tr>
    <tr>
      <th>178</th>
      <td>Hidden</td>
      <td>1920</td>
      <td>OU Detroit Scholarship</td>
      <td>ACPT</td>
      <td>472.0</td>
      <td>472.0</td>
      <td>Federal Pell Grant</td>
      <td>FDRL</td>
      <td>GRNT</td>
      <td>ACPT</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>9432</td>
      <td>13346</td>
    </tr>
  </tbody>
</table>
</div>



That does it for this Notebook! This is a prime example of how Jupyter Notebooks can be useful tools to assist you in day-to-day analytics type work. It makes it incredibly easy to reproduce your work and to write shareable code. If you have any feedback and/or questions, please don't hesitate to reach out! Thank you for taking the time to read this post, and take care of yourself!
