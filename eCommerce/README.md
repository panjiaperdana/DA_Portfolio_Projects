💻 Connecting to the database
- Import library
- Chekcing data


```python
import pandas as pd
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

engine = create_engine("postgresql://postgres:admin@localhost:5432/EDA_Portfolio_Projects")

customers = pd.read_sql("select * from ecomm_customers", engine)
products = pd.read_sql("select * from ecomm_products", engine)
orders = pd.read_sql("select * from ecomm_orders", engine)

df = orders.merge(customers, on="customer_id").merge(products, on="product_id")

print(df)
```

         order_id  customer_id  product_id  order_date  quantity     status  \
    0        1001            4         110  2023-10-01         9  cancelled   
    1        1002            4         101  2024-03-10         3  cancelled   
    2        1003            3         103  2023-08-06         2  completed   
    3        1004           16         108  2023-08-11         8  cancelled   
    4        1005            1         108  2023-11-15         2  cancelled   
    ..        ...          ...         ...         ...       ...        ...   
    495      1496           18         108  2023-09-13         7  completed   
    496      1497           12         105  2023-06-16         8  completed   
    497      1498            4         110  2024-05-18         9  completed   
    498      1499           17         102  2023-11-03         7  cancelled   
    499      1500            5         105  2024-01-10         4  completed   
    
                  name                email signup_date region    product_name  \
    0     Diana Prince    diana@example.com  2023-04-10   West         Pen Set   
    1     Diana Prince    diana@example.com  2023-04-10   West   Laptop Pro 15   
    2      Charlie Lee  charlie@example.com  2023-03-05   East    Office Chair   
    3      Paula Green    paula@example.com  2023-11-05   West   Standing Desk   
    4    Alice Johnson    alice@example.com  2023-01-15  North   Standing Desk   
    ..             ...                  ...         ...    ...             ...   
    495   Rachel Young   rachel@example.com  2023-11-20  South   Standing Desk   
    496    Linda Moore    linda@example.com  2023-09-15   West   Notebook Pack   
    497   Diana Prince    diana@example.com  2023-04-10   West         Pen Set   
    498     Quinn Hall    quinn@example.com  2023-11-15  North  Wireless Mouse   
    499    Ethan Brown    ethan@example.com  2023-05-25  North   Notebook Pack   
    
            category  price  
    0     Stationery      8  
    1    Electronics   1200  
    2      Furniture    150  
    3      Furniture    400  
    4      Furniture    400  
    ..           ...    ...  
    495    Furniture    400  
    496   Stationery      5  
    497   Stationery      8  
    498  Electronics     25  
    499   Stationery      5  
    
    [500 rows x 13 columns]
    

🔎 Step 1. Load and inspect the data
- Load data from PostgreSQL into Pandas.
- Check structure: .head(), .info(), .describe().
- Validate joins: Merge orders with customers and products to create a full dataset.


```python
df.head(10)
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
      <th>order_id</th>
      <th>customer_id</th>
      <th>product_id</th>
      <th>order_date</th>
      <th>quantity</th>
      <th>status</th>
      <th>name</th>
      <th>email</th>
      <th>signup_date</th>
      <th>region</th>
      <th>product_name</th>
      <th>category</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001</td>
      <td>4</td>
      <td>110</td>
      <td>2023-10-01</td>
      <td>9</td>
      <td>cancelled</td>
      <td>Diana Prince</td>
      <td>diana@example.com</td>
      <td>2023-04-10</td>
      <td>West</td>
      <td>Pen Set</td>
      <td>Stationery</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002</td>
      <td>4</td>
      <td>101</td>
      <td>2024-03-10</td>
      <td>3</td>
      <td>cancelled</td>
      <td>Diana Prince</td>
      <td>diana@example.com</td>
      <td>2023-04-10</td>
      <td>West</td>
      <td>Laptop Pro 15</td>
      <td>Electronics</td>
      <td>1200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1003</td>
      <td>3</td>
      <td>103</td>
      <td>2023-08-06</td>
      <td>2</td>
      <td>completed</td>
      <td>Charlie Lee</td>
      <td>charlie@example.com</td>
      <td>2023-03-05</td>
      <td>East</td>
      <td>Office Chair</td>
      <td>Furniture</td>
      <td>150</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1004</td>
      <td>16</td>
      <td>108</td>
      <td>2023-08-11</td>
      <td>8</td>
      <td>cancelled</td>
      <td>Paula Green</td>
      <td>paula@example.com</td>
      <td>2023-11-05</td>
      <td>West</td>
      <td>Standing Desk</td>
      <td>Furniture</td>
      <td>400</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1005</td>
      <td>1</td>
      <td>108</td>
      <td>2023-11-15</td>
      <td>2</td>
      <td>cancelled</td>
      <td>Alice Johnson</td>
      <td>alice@example.com</td>
      <td>2023-01-15</td>
      <td>North</td>
      <td>Standing Desk</td>
      <td>Furniture</td>
      <td>400</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1006</td>
      <td>7</td>
      <td>103</td>
      <td>2023-09-04</td>
      <td>9</td>
      <td>cancelled</td>
      <td>George Miller</td>
      <td>george@example.com</td>
      <td>2023-07-01</td>
      <td>East</td>
      <td>Office Chair</td>
      <td>Furniture</td>
      <td>150</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1007</td>
      <td>14</td>
      <td>109</td>
      <td>2024-02-29</td>
      <td>8</td>
      <td>completed</td>
      <td>Nina Lopez</td>
      <td>nina@example.com</td>
      <td>2023-10-12</td>
      <td>South</td>
      <td>Water Bottle</td>
      <td>Grocery</td>
      <td>10</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1008</td>
      <td>5</td>
      <td>104</td>
      <td>2023-07-23</td>
      <td>5</td>
      <td>completed</td>
      <td>Ethan Brown</td>
      <td>ethan@example.com</td>
      <td>2023-05-25</td>
      <td>North</td>
      <td>Coffee Beans 1kg</td>
      <td>Grocery</td>
      <td>12</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1009</td>
      <td>14</td>
      <td>110</td>
      <td>2023-10-29</td>
      <td>3</td>
      <td>completed</td>
      <td>Nina Lopez</td>
      <td>nina@example.com</td>
      <td>2023-10-12</td>
      <td>South</td>
      <td>Pen Set</td>
      <td>Stationery</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1010</td>
      <td>12</td>
      <td>101</td>
      <td>2023-08-22</td>
      <td>6</td>
      <td>returned</td>
      <td>Linda Moore</td>
      <td>linda@example.com</td>
      <td>2023-09-15</td>
      <td>West</td>
      <td>Laptop Pro 15</td>
      <td>Electronics</td>
      <td>1200</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 500 entries, 0 to 499
    Data columns (total 13 columns):
     #   Column        Non-Null Count  Dtype 
    ---  ------        --------------  ----- 
     0   order_id      500 non-null    int64 
     1   customer_id   500 non-null    int64 
     2   product_id    500 non-null    int64 
     3   order_date    500 non-null    object
     4   quantity      500 non-null    int64 
     5   status        500 non-null    object
     6   name          500 non-null    object
     7   email         500 non-null    object
     8   signup_date   500 non-null    object
     9   region        500 non-null    object
     10  product_name  500 non-null    object
     11  category      500 non-null    object
     12  price         500 non-null    int64 
    dtypes: int64(5), object(8)
    memory usage: 50.9+ KB
    


```python
df.describe()
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
      <th>order_id</th>
      <th>customer_id</th>
      <th>product_id</th>
      <th>quantity</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.00000</td>
      <td>500.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1250.500000</td>
      <td>10.846000</td>
      <td>105.700000</td>
      <td>5.44200</td>
      <td>250.238000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>144.481833</td>
      <td>5.867123</td>
      <td>2.875158</td>
      <td>2.78966</td>
      <td>363.404964</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1001.000000</td>
      <td>1.000000</td>
      <td>101.000000</td>
      <td>1.00000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1125.750000</td>
      <td>6.000000</td>
      <td>103.000000</td>
      <td>3.00000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1250.500000</td>
      <td>11.000000</td>
      <td>106.000000</td>
      <td>5.00000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1375.250000</td>
      <td>16.000000</td>
      <td>108.000000</td>
      <td>8.00000</td>
      <td>400.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1500.000000</td>
      <td>20.000000</td>
      <td>110.000000</td>
      <td>10.00000</td>
      <td>1200.000000</td>
    </tr>
  </tbody>
</table>
</div>



🧹 Step 2. Data quality checks
- Missing values: df.isnull().sum()
- Duplicates: df.duplicated().sum()
- Data types: Ensure dates are datetime, numeric fields are numeric.
- Outliers: Look for extreme quantities or prices.


```python
df.isnull().sum()
```




    order_id        0
    customer_id     0
    product_id      0
    order_date      0
    quantity        0
    status          0
    name            0
    email           0
    signup_date     0
    region          0
    product_name    0
    category        0
    price           0
    dtype: int64




```python
df.duplicated().sum()
```




    np.int64(0)




```python
# Change order_date to datetime format
df['cus_order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
df['cus_order_date_year'] = df['cus_order_date'].dt.year
df['cus_order_date_q'] = df['cus_order_date'].dt.to_period('Q')
df['cus_order_date_month'] = df['cus_order_date'].dt.month
df['cus_order_date_day'] = df['cus_order_date'].dt.day
df['cus_order_date_week'] = df['cus_order_date'].dt.isocalendar().week
df['cus_order_date_day_name'] = df['cus_order_date'].dt.strftime('%a')

# Change singup_date to date format
df['cus_signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')

# Categorical column to reduce memory usage and speed up grouping/aggregation
df['cus_status'] = df['status'].astype('category')
df['cus_region'] = df['region'].astype('category')
df['cus_product_name'] = df['product_name'].astype('category')
df['cus_category'] = df['category'].astype('category')

# Convert price to float
df['cus_price'] = df['price'].astype('float')

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 500 entries, 0 to 499
    Data columns (total 26 columns):
     #   Column                   Non-Null Count  Dtype         
    ---  ------                   --------------  -----         
     0   order_id                 500 non-null    int64         
     1   customer_id              500 non-null    int64         
     2   product_id               500 non-null    int64         
     3   order_date               500 non-null    object        
     4   quantity                 500 non-null    int64         
     5   status                   500 non-null    object        
     6   name                     500 non-null    object        
     7   email                    500 non-null    object        
     8   signup_date              500 non-null    object        
     9   region                   500 non-null    object        
     10  product_name             500 non-null    object        
     11  category                 500 non-null    object        
     12  price                    500 non-null    int64         
     13  cus_order_date           500 non-null    datetime64[ns]
     14  cus_order_date_year      500 non-null    int32         
     15  cus_order_date_q         500 non-null    period[Q-DEC] 
     16  cus_order_date_month     500 non-null    int32         
     17  cus_order_date_day       500 non-null    int32         
     18  cus_order_date_week      500 non-null    UInt32        
     19  cus_order_date_day_name  500 non-null    object        
     20  cus_signup_date          500 non-null    datetime64[ns]
     21  cus_status               500 non-null    category      
     22  cus_region               500 non-null    category      
     23  cus_product_name         500 non-null    category      
     24  cus_category             500 non-null    category      
     25  cus_price                500 non-null    float64       
    dtypes: UInt32(1), category(4), datetime64[ns](2), float64(1), int32(3), int64(5), object(9), period[Q-DEC](1)
    memory usage: 81.6+ KB
    

📊 Step 3. Descriptive statistics
- Orders: Count, min/max dates, average quantity.
- Revenue: quantity * price → total revenue per order.
- Customers: Number of unique customers, average orders per customer.
- Products: Top‑selling products, average price per category.


```python
# Count of Orders
order_count = len(df)
# Min/Max Dates
min_date = df['cus_order_date'].min()
max_date = df['cus_order_date'].max()
# Average Quantity
avg_qty = orders['quantity'].mean()

print("Orders:", order_count)
print("Date Range:", min_date, "to", max_date)
print("Avg Quantity:", avg_qty)
```

    Orders: 500
    Date Range: 2023-06-01 00:00:00 to 2024-05-31 00:00:00
    Avg Quantity: 5.442
    


```python
# Revenue
df['revenue'] = df['quantity'] * df['cus_price']
# Total Revenue
total_revenue = df['revenue'].sum()
# Avg Revenue per Order
avg_revenue_per_order = df['revenue'].mean()

print("Total Revenue:", total_revenue)
print("Average Revenue:", avg_revenue_per_order)
```

    Total Revenue: 657708.0
    Average Revenue: 1315.416
    


```python
# Unique Customer
unique_cust = df['customer_id'].nunique()
# Avg Order per Customer
avg_order_per_cust = order_count/unique_cust
# Top 5 Customer
top_customer = df.groupby("name")['revenue'].sum().sort_values(ascending=False).head(5)

print("Unique Customers:", unique_cust)
print("Average Orders per Customer:", avg_order_per_cust)
print("Top Customers:", top_customer)
```

    Unique Customers: 20
    Average Orders per Customer: 25.0
    Top Customers: name
    Paula Green      58745.0
    George Miller    50887.0
    Diana Prince     47751.0
    Rachel Young     47688.0
    Helen Clark      45454.0
    Name: revenue, dtype: float64
    


```python
# Top 5 Products
top_products = df.groupby("product_name")['revenue'].sum().sort_values(ascending=False).head(5)
# Avg Price per Category
avg_price_per_categ = df.groupby('category')['cus_price'].mean()

print("Top Products:", top_products)
print("Average Price:", avg_price_per_categ)
```

    Top Products: product_name
    Laptop Pro 15             253200.0
    Smartphone X              183200.0
    Standing Desk             119200.0
    Office Chair               43500.0
    Headphones NoiseCancel     41400.0
    Name: revenue, dtype: float64
    Average Price: category
    Electronics    522.457627
    Furniture      287.850467
    Grocery         11.020833
    Stationery       6.550000
    Name: cus_price, dtype: float64
    

📈 Step 4. Visual exploration
Use Matplotlib/Seaborn for plots:
- Distribution plots: Quantity, revenue.
- Bar charts: Revenue by category, orders by region.
- Line charts: Daily/Monthly revenue trends.
- Pie chart: Status breakdown (completed vs cancelled vs returned).


```python
# Quantity Distribution
sns.histplot(df['quantity'], bins=10, kde=True)
plt.title("Distribution of Order Quantity")
plt.xlabel("Quantity")
plt.ylabel('Frequency')
plt.show()
```


    
![png](EDA_eCommerce_files/EDA_eCommerce_16_0.png)
    



```python
# Revenue Distribution
sns.histplot(df['revenue'], bins=20, kde=True, color='green')
plt.title("Distribution of Order Revenue")
plt.xlabel("Revenue")
plt.ylabel('Frequency')
plt.show()
```


    
![png](EDA_eCommerce_files/EDA_eCommerce_17_0.png)
    



```python
# Revenue by Category
cat_rev = df.groupby('cus_category', observed=True)['revenue'].sum().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='cus_category', y='revenue', data=cat_rev, palette="Blues_d", hue='cus_category')
plt.title("Revenue by Product Category")
plt.xticks(rotation=45)
plt.show()
```


    
![png](EDA_eCommerce_files/EDA_eCommerce_18_0.png)
    



```python
# Orders by Region
region_orders = df.groupby('cus_region', observed=True)['order_id'].count().reset_index()
sns.barplot(x='cus_region', y='order_id', hue='cus_region', data=region_orders, palette="Oranges_r")
plt.title('Orders by Region')
plt.show()
```


    
![png](EDA_eCommerce_files/EDA_eCommerce_19_0.png)
    



```python
# Daily Revenue Trend
daily_rev = df.groupby('cus_order_date')['revenue'].sum().reset_index()
plt.figure(figsize=(10, 5))
sns.lineplot(x='cus_order_date', y='revenue', data=daily_rev)
plt.title('Daily Revenue Trend')
plt.xticks(rotation=45)
plt.show()
```


    
![png](EDA_eCommerce_files/EDA_eCommerce_20_0.png)
    



```python
# Monthly Revenue Trend
monthly_rev = df.groupby('cus_order_date_month')['revenue'].sum().reset_index()
sns.lineplot(x='cus_order_date_month', y='revenue', data=monthly_rev, marker='o')
plt.title('Monthly Revenue Trend')
plt.xticks(rotation=45)
plt.show()
```


    
![png](EDA_eCommerce_files/EDA_eCommerce_21_0.png)
    



```python
# Status Breakdown
status_counts = df['cus_status'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(status_counts, labels=status_counts.index, autopct="%1.1f%%", startangle=90, colors=['red', 'green', 'blue'])
plt.title("Order Status Breakdown")
plt.show()
```


    
![png](EDA_eCommerce_files/EDA_eCommerce_22_0.png)
    


🧩 Step 5. Customer analysis
- Retention: First vs last order date per customer.
- Segmentation: Revenue by region.
- Top customers: Sort by total spend.


```python
# First and Last Order per Customer
retention = df.groupby('customer_id').agg(
    first_order = ('cus_order_date', 'min'),
    last_order = ('cus_order_date', 'max'),
    total_order = ('order_id', 'count'),
    total_revenue = ('revenue', 'sum'),
).reset_index()
# Calculate Active Days
retention['active_days'] = (retention['last_order'] - retention['first_order']).dt.days

print(retention.head())
```

       customer_id first_order last_order  total_order  total_revenue  active_days
    0            1  2023-06-01 2024-05-29           19        44088.0          363
    1            2  2023-06-04 2024-05-08           27        38356.0          339
    2            3  2023-06-05 2024-05-18           27        43557.0          348
    3            4  2023-07-01 2024-05-20           26        47751.0          324
    4            5  2023-06-17 2024-05-29           19        13846.0          347
    


```python
# Revenue by Region
region_rev = df.groupby('cus_region', observed=True)['revenue'].sum().reset_index().sort_values('revenue', ascending=False)

print(region_rev)
```

      cus_region   revenue
    3       West  200537.0
    2      South  176956.0
    0       East  157332.0
    1      North  122883.0
    


```python
# Top Customers - Total Spend
top_customers = df.groupby('name')['revenue'].sum().reset_index().sort_values('revenue', ascending=False).head(10)

print(top_customers)
```

                 name  revenue
    15    Paula Green  58745.0
    6   George Miller  50887.0
    3    Diana Prince  47751.0
    17   Rachel Young  47688.0
    7     Helen Clark  45454.0
    0   Alice Johnson  44088.0
    2     Charlie Lee  43557.0
    13     Nina Lopez  39125.0
    1       Bob Smith  38356.0
    18     Sam Carter  34776.0
    

🪄 Step 6. Product analysis
- Best sellers: Highest revenue products.
- Category trends: Compare Electronics vs Furniture vs Grocery.
- Price sensitivity: Correlate product price with quantity sold.


```python
# Top 10 Products
best_sellers = df.groupby('cus_product_name', observed=True)['revenue'].sum().reset_index().sort_values('revenue', ascending=False).head(10)

print(best_sellers)
```

             cus_product_name   revenue
    2           Laptop Pro 15  253200.0
    6            Smartphone X  183200.0
    7           Standing Desk  119200.0
    4            Office Chair   43500.0
    1  Headphones NoiseCancel   41400.0
    9          Wireless Mouse    6875.0
    0        Coffee Beans 1kg    3108.0
    8            Water Bottle    2920.0
    5                 Pen Set    2680.0
    3           Notebook Pack    1625.0
    


```python
# Revenue by Category Over Time
cat_trends = df.groupby(['cus_order_date_month', 'cus_category'], observed=True)['revenue'].sum().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x='cus_order_date_month', y='revenue', hue='cus_category', data=cat_trends, marker='o')
plt.title('Monthly Revenue Trends by Category')
plt.xticks(rotation=45)
plt.show()
```


    
![png](EDA_eCommerce_files/EDA_eCommerce_29_0.png)
    



```python
# Total Quantity Sold per Product
qty_by_product = df.groupby(['cus_product_name', 'cus_price'], observed=True)['quantity'].sum().reset_index()
# Scatter Plot: Price vs Qty Sold
plt.figure(figsize=(8, 6))
sns.scatterplot(x='cus_price', y='quantity', hue='cus_product_name', data=qty_by_product, s=100)
plt.title('Price vs Quantity Sold (Price Sensitivity)')
plt.xlabel("Product Price")
plt.ylabel("Total Quantity Sold")
plt.show()
```


    
![png](EDA_eCommerce_files/EDA_eCommerce_30_0.png)
    


📅 Step 7. Time series analysis
- Seasonality: Monthly revenue trends.
- Growth: Compare year‑over‑year or quarter‑over‑quarter.
- Peak days: Which days have highest sales?


```python
# Monthly Revenue
sns.lineplot(x='cus_order_date_month', y='revenue', data=monthly_rev, marker='o')
plt.title('Monthly Revenue Trend')
plt.xticks(rotation=45)
plt.show()
```


    
![png](EDA_eCommerce_files/EDA_eCommerce_32_0.png)
    



```python
# Yearly Revenue
yearly_rev = df.groupby('cus_order_date_year')['revenue'].sum().reset_index()
yearly_rev['YoY_growth'] = yearly_rev['revenue'].pct_change() * 100

print(yearly_rev)
```

       cus_order_date_year   revenue  YoY_growth
    0                 2023  449267.0         NaN
    1                 2024  208441.0  -53.604204
    


```python
quarterly_rev = df.groupby('cus_order_date_q')['revenue'].sum().reset_index()
quarterly_rev['QoQ_Growth'] = quarterly_rev['revenue'].pct_change() * 100

print(quarterly_rev)
```

      cus_order_date_q   revenue  QoQ_Growth
    0           2023Q2   43947.0         NaN
    1           2023Q3  184855.0  320.631670
    2           2023Q4  220465.0   19.263747
    3           2024Q1  151954.0  -31.075681
    4           2024Q2   56487.0  -62.826250
    


```python
# Top 10 Peak Days
peak_days = daily_rev.sort_values('revenue', ascending=False).head(10)

print(peak_days)
```

        cus_order_date  revenue
    120     2023-11-07  22424.0
    183     2024-02-02  18600.0
    25      2023-07-05  16816.0
    117     2023-11-04  13150.0
    214     2024-03-13  12908.0
    135     2023-11-29  12050.0
    86      2023-09-25  12000.0
    122     2023-11-10  11800.0
    181     2024-01-31  11000.0
    194     2024-02-19  10970.0
    

✅ Step 8. Summarize insights
- Which region drives most revenue?
- Which product category is most profitable?
- What’s the return/cancellation rate?
- Who are the top 5 customers?
- Are there seasonal spikes in sales?


```python
# Summarize
print('Region Revenue:')
print(region_rev)
print('=========================================================')
category_rev = df.groupby('cus_category', observed=True)['revenue'].sum().reset_index().sort_values('revenue', ascending=False)
print('Profitable Products by Category:')
print(category_rev)
print('=========================================================')
status_rate = status_counts / status_counts.sum() * 100
print('Status Rate:')
print(status_rate)
print('=========================================================')
print('Top Customers:')
print(top_customers)
print('=========================================================')
print('Monthly Revenue:')
sns.lineplot(x='cus_order_date_month', y='revenue', data=monthly_rev, marker='o')
plt.title('Monthly Revenue Trend')
plt.xticks(rotation=45)
plt.show()
```

    Region Revenue:
      cus_region   revenue
    3       West  200537.0
    2      South  176956.0
    0       East  157332.0
    1      North  122883.0
    =========================================================
    Profitable Products by Category:
      cus_category   revenue
    0  Electronics  484675.0
    1    Furniture  162700.0
    2      Grocery    6028.0
    3   Stationery    4305.0
    =========================================================
    Status Rate:
    cus_status
    completed    71.4
    cancelled    19.2
    returned      9.4
    Name: count, dtype: float64
    =========================================================
    Top Customers:
                 name  revenue
    15    Paula Green  58745.0
    6   George Miller  50887.0
    3    Diana Prince  47751.0
    17   Rachel Young  47688.0
    7     Helen Clark  45454.0
    0   Alice Johnson  44088.0
    2     Charlie Lee  43557.0
    13     Nina Lopez  39125.0
    1       Bob Smith  38356.0
    18     Sam Carter  34776.0
    =========================================================
    Monthly Revenue:
    


    
![png](EDA_eCommerce_files/EDA_eCommerce_37_1.png)
    


🤖 Implement Machine Learning

🗂 1. Sales Forecasting
Forecasted revenue with confidence intervals.


```python
# DB connection
engine = create_engine("postgresql+psycopg2://postgres:admin@localhost:5432/EDA_Portfolio_Projects")

# Daily revenue
daily_rev = df.groupby('cus_order_date')['revenue'].sum().reset_index()
daily_rev['cus_order_date'] = pd.to_datetime(daily_rev['cus_order_date'])
daily_rev = daily_rev.set_index('cus_order_date').asfreq('D', fill_value=0)

# Train SARIMA
model = SARIMAX(daily_rev['revenue'], order=(1,1,1), seasonal_order=(1,1,1,7))
res = model.fit(disp=False)

# Forecast next 90 days
future_fc = res.get_forecast(steps=90)
future_mean = future_fc.predicted_mean
future_ci = future_fc.conf_int()

forecast_df = pd.DataFrame({
    "date": future_mean.index,
    "forecast_revenue": future_mean.values,
    "lower_ci": future_ci.iloc[:,0].values,
    "upper_ci": future_ci.iloc[:,1].values
})

# Visualization
plt.figure(figsize=(12,6))
plt.plot(daily_rev.index, daily_rev['revenue'], label="Actual")
plt.plot(future_mean.index, future_mean, label="Forecast", color="green")
plt.fill_between(future_ci.index, future_ci.iloc[:,0], future_ci.iloc[:,1], color="green", alpha=0.2)
plt.title("Daily Revenue Forecast (Next 90 Days)")
plt.legend()
plt.show()

# Insert into DB
forecast_df.to_sql("ecomm_forecast_daily", engine, if_exists="replace", index=False)
```


    
![png](EDA_eCommerce_files/EDA_eCommerce_40_0.png)
    





    90



🛒 2. Product Recommendation
Customer → Recommended products.


```python
# Customer-product matrix
cust_prod_matrix = df.pivot_table(
    index="customer_id",
    columns="product_name",
    values="quantity",
    aggfunc="sum",
    fill_value=0
)

# Cosine similarity
similarity = cosine_similarity(cust_prod_matrix)
similarity_df = pd.DataFrame(similarity, index=cust_prod_matrix.index, columns=cust_prod_matrix.index)

# Heatmap visualization
plt.figure(figsize=(10,8))
sns.heatmap(similarity_df, cmap="coolwarm", annot=False)
plt.title("Customer Similarity Heatmap (Cosine)")
plt.xlabel("Customer ID")
plt.ylabel("Customer ID")
plt.show()

# Generate recommendations
recommendations = []
for cust in cust_prod_matrix.index:
    sim_scores = similarity_df.loc[cust].sort_values(ascending=False).drop(cust)
    top_customers = sim_scores.head(5).index
    similar_purchases = cust_prod_matrix.loc[top_customers].sum()
    already_bought = cust_prod_matrix.loc[cust]
    recs = similar_purchases[already_bought == 0].sort_values(ascending=False).head(3)
    for product in recs.index:
        recommendations.append({
            "customer_id": cust,
            "recommended_product": product,
            "reason": "Similar customers purchased this"
        })

recommendations_df = pd.DataFrame(recommendations)

# Insert into DB
recommendations_df.to_sql("ecomm_recommendations", engine, if_exists="replace", index=False)
```


    
![png](EDA_eCommerce_files/EDA_eCommerce_42_0.png)
    





    25



📦 3. Order Status Prediction
Actual vs Predicted order status with probabilities.


```python
# Features & target
features = df[["quantity","cus_price","revenue","cus_region","cus_category","cus_order_date_day_name","cus_order_date_month"]]
target = df["cus_status"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

# Preprocessing
categorical_cols = ["cus_region","cus_category","cus_order_date_day_name","cus_order_date_month"]
numeric_cols = ["quantity","cus_price","revenue"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols)
])

# Model
clf = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title("Confusion Matrix - Order Status Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(y_test, y_pred))

# Build predictions table
predictions_df = pd.DataFrame({
    "order_id": X_test.index,
    "customer_id": df.loc[X_test.index, "customer_id"].values,
    "actual_status": y_test.values,
    "predicted_status": y_pred,
    "probability_completed": y_proba[:, list(clf.classes_).index("completed")],
    "probability_cancelled": y_proba[:, list(clf.classes_).index("cancelled")],
    "probability_returned": y_proba[:, list(clf.classes_).index("returned")]
})

# Insert into DB
predictions_df.to_sql("ecomm_order_predictions", engine, if_exists="replace", index=False)
```


    
![png](EDA_eCommerce_files/EDA_eCommerce_44_0.png)
    


                  precision    recall  f1-score   support
    
       cancelled       0.50      0.11      0.17        19
       completed       0.73      0.96      0.83        72
        returned       0.00      0.00      0.00         9
    
        accuracy                           0.71       100
       macro avg       0.41      0.35      0.33       100
    weighted avg       0.62      0.71      0.63       100
    
    




    100


