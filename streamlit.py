import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sqlalchemy import create_engine, text
import matplotlib.ticker as ticker

# SQLite database file path
db_path = "./analysis.db"

# Database connection
engine = create_engine(f"sqlite:///{db_path}")

# Streamlit app
def main():
    st.title("Data Exploration App")

    # SQL query for the "Sales_Fact" table
    sales_query = "SELECT * FROM Sales_Fact"
    sales_df = pd.read_sql_query(sales_query, engine)

    # Display the Sales_Fact data
    st.header("Sales_Fact Data")
    st.dataframe(sales_df)

    # SQL query for the "Store" table
    store_query = "SELECT * FROM Store"
    store_df = pd.read_sql_query(store_query, engine)

    # Display the Store data
    st.header("Store Data")
    st.dataframe(store_df)

    # SQL query for the "Product" table
    product_query = "SELECT * FROM Product"
    product_df = pd.read_sql_query(product_query, engine)

    # Display the Product data
    st.header("Product Data")
    st.dataframe(product_df)

    # SQL query for the "Date_Dimension" table
    date_query = "SELECT * FROM Date_Dimension"
    date_df = pd.read_sql_query(date_query, engine)

    # Display the Date_Dimension data
    st.header("Date_Dimension Data")
    st.dataframe(date_df)

    # SQL query for creating another temporary table
    create_temp_table_query = text("""
    CREATE TEMPORARY TABLE AnotherTempTable AS
    SELECT
        s.Sale_ID,
        s.Date,
        s.Store_ID,
        s.Product_ID,
        s.Units_Sold,
        s.Total_Cost,
        s.Total_Sale,
        CAST(s.Product_Cost AS FLOAT) AS Product_Cost,
        CAST(s.Product_Price AS FLOAT) AS Product_Price,
        p.Product_Name,
        p.Product_Category,
        st.Store_Name,
        st.Store_City,
        st.Store_Location
    FROM 
        Sales_Fact s
        JOIN Product p ON s.Product_ID = p.Product_ID
        JOIN Store st ON s.Store_ID = st.Store_ID
    """)

    # Execute the query to create another temporary table
    with engine.connect() as connection:
        connection.execute(create_temp_table_query)

    # SQL query to load data from the new temporary table into a pandas DataFrame
    another_temp_table_df = pd.read_sql_query("SELECT * FROM AnotherTempTable", engine)

    # Display the data from the new temporary table
    st.header("Another Temp Table Data")
    st.dataframe(another_temp_table_df)

    # SQL query to load data from TempSales2 into a pandas DataFrame
    query2 = "SELECT * FROM AnotherTempTable"
    sales2_df = pd.read_sql_query(query2, engine)


    # SQL query for the plot
    query_plot = """
    SELECT 
        Store_City, 
        Product_Category, 
        SUM(Total_Sale) AS Total_Sale
    FROM 
        AnotherTempTable
    GROUP BY 
        Store_City, 
        Product_Category;
    """

    # Execute the SQL query and store the result in a DataFrame
    data_for_plot = pd.read_sql_query(query_plot, engine)

    # Plotting
    plt.figure(figsize=(20, 10))
    barplot = sns.barplot(
        data=data_for_plot,
        x='Product_Category',
        y='Total_Sale',
        hue='Store_City',
        palette='deep'
    )

    plt.legend(title='Store Cities', loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45, ha='right')

    # Annotating the bars
    for p in barplot.patches:
        height = p.get_height()
        if not pd.isna(height): 
            barplot.annotate(f'{height:,.0f}',
                             (p.get_x() + p.get_width() / 2, height),
                             ha='center', va='bottom',
                             xytext=(0, 4),
                             textcoords='offset points')

    plt.xlabel('Product Category')
    plt.ylabel('Sales')
    plt.title('Sales by Product Category and Store City')

    # Display the plot in Streamlit
    st.header("Sales by Product Category and Store City")
    st.pyplot(plt)

    # Calculate and display average daily sales per store
    sales4 = sales2_df.copy()
    sales4['Date'] = pd.to_datetime(sales2_df['Date'])

    store_dates = sales4.groupby('Store_Name')['Date'].agg(['min', 'max']).reset_index()
    store_dates['Operating_Days'] = (store_dates['max'] - store_dates['min']).dt.days + 1

    store_sales_counts = sales4.groupby('Store_Name')['Sale_ID'].count().sort_values(ascending=False).reset_index(name='Total_Number_of_Sales')

    store_sales_info = pd.merge(store_sales_counts, store_dates, on='Store_Name')
    store_sales_info['Average_Daily_Sales'] = store_sales_info['Total_Number_of_Sales'] / store_sales_info['Operating_Days']

    sns.set(style="whitegrid")
    colormap = plt.cm.hsv
    colors = colormap(np.linspace(0, 1, len(store_sales_info['Store_Name'])))

    plt.figure(figsize=(25, 10))
    bars = plt.bar(store_sales_info['Store_Name'], store_sales_info['Average_Daily_Sales'], color=colors)

    plt.xlabel('Store Name', fontsize=12)
    plt.ylabel('Average Daily Sales', fontsize=12)
    plt.title('Average Daily Sales per Store', fontsize=16)

    plt.xticks(ticks=range(len(store_sales_info['Store_Name'])), labels=store_sales_info['Store_Name'].tolist(), rotation=45, ha='right')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), ha='center', va='bottom')

    plt.tight_layout()
    st.header("Average Daily Sales per Store")
    st.pyplot(plt)


   # Calculate and display total number of sales per store
    store_dates = sales2_df.groupby('Store_Name')['Date'].agg([min, max]).reset_index()
    store_dates['min'] = pd.to_datetime(store_dates['min'])  # Dönüştürme eklendi
    store_dates['max'] = pd.to_datetime(store_dates['max'])  # Dönüştürme eklendi
    store_dates['Operating_Days'] = (store_dates['max'] - store_dates['min']).dt.days + 1

    store_sales_counts = sales2_df.groupby('Store_Name')['Sale_ID'].count().sort_values(ascending=False).reset_index(name='Total_Number_of_Sales')
    store_sales_info = pd.merge(store_sales_counts, store_dates, on='Store_Name')

    sns.set(style="whitegrid")
    colormap = plt.cm.hsv
    colors = colormap(np.linspace(0, 1, len(store_sales_info['Store_Name'])))

    plt.figure(figsize=(30, 10))
    bars = plt.bar(store_sales_info['Store_Name'], store_sales_info['Total_Number_of_Sales'], color=colors)

    plt.xlabel('Store Name', fontsize=12)
    plt.ylabel('Total Number Of Sales', fontsize=12)
    plt.title('Total Number Of Sales per Store', fontsize=16)

    plt.xticks(ticks=range(len(store_sales_info['Store_Name'])), labels=store_sales_info['Store_Name'].tolist(), rotation=45, ha='right')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), ha='center', va='bottom')

    plt.tight_layout()

    # Display the plot in Streamlit
    st.header("Total Number Of Sales per Store")
    st.pyplot(plt)


    # Calculate and display best selling product count per store
    product_sales_counts = sales2_df.groupby(['Store_Name', 'Product_Name'])['Sale_ID'].count().reset_index(name='Number_of_Sales')
    best_selling_products = product_sales_counts.loc[product_sales_counts.groupby('Store_Name')['Number_of_Sales'].idxmax()]
    best_selling_products = pd.merge(best_selling_products, store_dates[['Store_Name', 'Operating_Days']], on='Store_Name')

    sns.set(style="whitegrid")
    colormap = plt.cm.hsv
    colors = colormap(np.linspace(0, 1, len(best_selling_products['Store_Name'])))

    plt.figure(figsize=(30, 10))
    bars = plt.bar(best_selling_products['Store_Name'], best_selling_products['Number_of_Sales'], color=colors)

    plt.xlabel('Store Name', fontsize=12)
    plt.ylabel('Best Selling Product Count', fontsize=12)
    plt.title('Best Selling Product Count per Store', fontsize=16)

    plt.xticks(ticks=range(len(best_selling_products['Store_Name'])), labels=best_selling_products['Store_Name'].tolist(), rotation=45, ha='right')

    for bar, product_name in zip(bars, best_selling_products['Product_Name']):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{product_name}: {int(yval)}", ha='center', va='bottom', rotation=90)

    plt.tight_layout()

    # Display the plot in Streamlit
    st.header("Best Selling Product Count per Store")
    st.pyplot(plt)


    # Calculate and display total sales per product
    total_product_sales = sales2_df.groupby(['Product_Name', 'Product_ID'])['Units_Sold'].sum().reset_index(name='Total_Sales')
    total_product_sales_sorted = total_product_sales.sort_values('Total_Sales', ascending=False)

    plt.figure(figsize=(30, 10))
    colors = plt.cm.hsv(np.linspace(0, 1, len(total_product_sales_sorted)))
    plt.bar(total_product_sales_sorted['Product_Name'], total_product_sales_sorted['Total_Sales'], color=colors)
    plt.xlabel('Product Name', fontsize=12)
    plt.ylabel('Total Sales', fontsize=12)
    plt.title('Total Sales per Product', fontsize=16)
    plt.xticks(rotation=90)

    # Display the plot in Streamlit
    st.header("Total Sales per Product")
    st.pyplot(plt)


    # Calculate and display total sales per product category
    product_category_sales = sales2_df.groupby(['Product_Category'])['Total_Sale'].sum().sort_values(ascending=False)

    # Plotting
    plt.figure(figsize=(15, 15))
    stacked_barplot = sales2_df.groupby(['Store_Name','Product_Category']).sum().unstack().plot(
        kind='barh',
        y='Total_Sale',
        stacked=True,
        figsize=(15, 15),
        colormap='viridis'
    )

    plt.xlabel('Total Sales', fontsize=12)
    plt.ylabel('Store Name', fontsize=12)
    plt.title('Total Sales per Store and Product Category', fontsize=16)

    # Display the plot in Streamlit
    st.header("Total Sales per Store and Product Category")
    st.pyplot(plt)



    # Plotting
    plt.figure(figsize=(8, 8))
    stacked_barplot = sales2_df.groupby(['Store_Location','Product_Category']).sum().unstack().plot(
        kind='bar',
        y='Total_Sale',
        stacked=True,
        figsize=(8, 8),
        colormap='viridis'
    )

    plt.xlabel('Store Location', fontsize=12)
    plt.ylabel('Total Sales', fontsize=12)
    plt.title('Total Sales per Store Location and Product Category', fontsize=16)

    # Display the plot in Streamlit
    st.header("Total Sales per Store Location and Product Category")
    st.pyplot(plt)


    # Plotting
    plt.figure(figsize=(8, 8))
    stacked_barplot = sales2_df.groupby(['Product_Name','Store_Location']).sum().unstack().plot(
        kind='barh',
        y='Total_Sale',
        stacked=True,
        figsize=(8, 8),
        colormap='viridis'
    )

    plt.xlabel('Total Sales', fontsize=12)
    plt.ylabel('Product Name', fontsize=12)
    plt.title('Total Sales per Product and Store Location', fontsize=16)
    plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    # Display the plot in Streamlit
    st.header("Total Sales per Product and Store Location")
    st.pyplot(plt)


    # Display total sales by city
    st.header("Total Sales by City")
    total_sales_by_city = sales2_df.groupby('Store_City')['Total_Sale'].sum().sort_values(ascending=False)
    st.write(total_sales_by_city)

    # Display highest sales by city for each product type
    st.header("City with Highest Sales for Each Toy Type")
    sales_by_product_city = sales2_df.groupby(['Product_Name', 'Store_City'])['Total_Sale'].sum().reset_index()
    idx = sales_by_product_city.groupby(['Product_Name'])['Total_Sale'].idxmax()
    highest_sales_by_city = sales_by_product_city.loc[idx]

    highest_sales_by_city = highest_sales_by_city.sort_values('Total_Sale', ascending=False)

    # Plotting
    plt.figure(figsize=(14, 10))
    barplot = sns.barplot(
        data=highest_sales_by_city,
        y='Product_Name',
        x='Total_Sale',
        hue='Store_City',
        dodge=False
    )

    plt.xlabel('Total Sales')
    plt.ylabel('Toy Type')
    plt.title('City with Highest Sales for Each Toy Type')
    plt.legend(title='City', bbox_to_anchor=(1.05, 1), loc='upper left')

    for p in barplot.patches:
        barplot.annotate(format(p.get_width(), '.0f'), 
                        (p.get_width(), p.get_y() + p.get_height() / 2),  
                        ha='left',  
                        va='center', 
                        xytext=(5, 0),  
                        textcoords='offset points')

    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(plt)


    # Calculate and display store age distribution
    st.header("Store Age Distribution")
    sales2_df['Store_Age'] = 2024 - store_df['Store_Open_Date_Year']

    sns.set_style('whitegrid')
    colormap = plt.cm.hsv
    colors = colormap(np.linspace(0, 1, len(sales2_df['Store_Age'].unique())))

    plt.figure(figsize=(14, 7))
    sns.countplot(x='Store_Age', data=sales2_df, palette=colors)

    plt.xlabel('Store Age')
    plt.ylabel('Count')
    plt.title('Store Age Distribution')

    # Display the plot in Streamlit
    st.pyplot(plt)


    # Display mean store age
    st.header("Mean Store Age")
    mean_store_age = sales2_df['Store_Age'].mean()
    st.write(f"The mean store age is: {mean_store_age:.2f} years")

    # Display total sales by product name
    st.header("Total Sales by Product Name")
    total_sales_by_product = sales2_df.groupby(['Product_Name'])['Total_Sale'].sum().sort_values(ascending=False)
    st.write(total_sales_by_product)


    # Display sales trend
    st.header("Sales Trend Over Time")
    sales2_df['Date'] = pd.to_datetime(sales2_df['Date'])
    sales_daily = sales2_df.groupby('Date')['Total_Sale'].sum()

    sales_daily_df = pd.DataFrame(sales_daily)
    sales_daily_df.index = pd.to_datetime(sales_daily_df.index)

    sales_weekly_mean = sales_daily_df['Total_Sale'].resample('W').mean()
    sales_monthly_mean = sales_daily_df['Total_Sale'].resample('M').mean()

    plt.figure(figsize=(30, 10))
    plt.plot(sales_daily, label='Daily')
    plt.plot(sales_weekly_mean, label='Weekly Mean')
    plt.plot(sales_monthly_mean, label='Monthly Mean')
    plt.title('Daily Weekly Monthly Sales Trend')
    plt.legend()
    st.pyplot(plt)



    # Daily Sales Plot
    sales2_df['Date'] = pd.to_datetime(sales2_df['Date'])
    sales_daily = sales2_df.groupby('Date')['Total_Sale'].sum()

    plt.figure(figsize=(30, 6))
    plt.plot(sales_daily, label='Daily Sales')
    plt.title('Daily Sales Trend')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.legend()
    st.pyplot(plt)

    # Monthly Sales Plot
    sales_monthly = sales2_df.groupby([sales2_df['Date'].dt.year.rename('year'), sales2_df['Date'].dt.to_period("M").rename('month')])['Total_Sale'].sum().reset_index()
    sales_monthly['month'] = sales_monthly['month'].astype(str)

    plt.figure(figsize=(30, 6))
    plt.plot(sales_monthly['month'], sales_monthly['Total_Sale'], label='Monthly Sales')
    plt.title('Monthly Sales Trend')
    plt.xlabel('Month')
    plt.ylabel('Total Sales')
    plt.legend()
    st.pyplot(plt)




    # Make a copy of sales2 to keep it unchanged
    product_info = sales2_df.copy()

    # Convert 'Product_Cost' and 'Product_Price' to floats
    product_info['Product_Cost'] = product_info['Product_Cost'].astype(float)
    product_info['Product_Price'] = product_info['Product_Price'].astype(float)

    # Calculate 'Product_Return' and 'Return_Rate'
    product_info['Product_Return'] = product_info['Product_Price'] - product_info['Product_Cost']
    product_info['Return_Rate'] = round(product_info['Product_Return'] / product_info['Product_Cost'], 3)

    # Sort by 'Return_Rate' in descending order and drop duplicates based on 'Product_ID'
    productnew_sorted = product_info.drop_duplicates(subset='Product_ID').sort_values(by='Return_Rate', ascending=False)

    # Displaying the result
    st.header("Product Information with Return Rate")
    st.dataframe(productnew_sorted[['Product_ID', 'Product_Name', 'Product_Category', 'Product_Cost', 'Product_Price', 'Product_Return', 'Return_Rate']])


    # Plotting
    colormap = plt.cm.hsv  
    colors = colormap(np.linspace(0, 1, len(productnew_sorted)))

    plt.figure(figsize=(14, 7))
    bars = plt.bar(productnew_sorted['Product_Name'], productnew_sorted['Return_Rate'], color=colors)

    plt.xlabel('Product Name', fontsize=12)
    plt.ylabel('Return Rate', fontsize=12)
    plt.title('Return Rate by Product', fontsize=16)

    plt.xticks(ticks=range(len(productnew_sorted['Product_Name'])), labels=productnew_sorted['Product_Name'].tolist(), rotation=45, ha='right')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

    plt.tight_layout()

    # Display the plot in Streamlit
    st.header("Return Rate by Product")
    st.pyplot(plt)



    # Plotting
    ProfitTable = pd.merge(productnew_sorted, total_product_sales_sorted, on='Product_Name')
    ProfitTable['Profit'] = ProfitTable['Product_Return'] * ProfitTable['Total_Sales']
    sorted_profit_table = ProfitTable[['Product_Name', 'Profit']].sort_values(by='Profit', ascending=False)
    colormap = plt.cm.viridis
    colors = colormap(np.linspace(0, 1, len(sorted_profit_table)))

    plt.figure(figsize=(30, 10))
    bars = plt.bar(sorted_profit_table['Product_Name'], sorted_profit_table['Profit'], color=colors)

    plt.xlabel('Product Name', fontsize=12)
    plt.ylabel('Profit', fontsize=12)
    plt.title('Profit by Product', fontsize=16)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.0f}', ha='center', va='bottom')

    plt.xticks(rotation=45, ha='right')

    # Display the plot in Streamlit
    st.header("Profit by Product")
    st.pyplot(plt)

    # Execute the query
    highest_cost_query = '''
    SELECT p.Product_Name, AVG(sf.Product_Cost) as Average_Cost
    FROM Sales_Fact sf
    JOIN Product p ON sf.Product_ID = p.Product_ID
    GROUP BY sf.Product_ID
    ORDER BY Average_Cost DESC
    LIMIT 5
    '''
    highest_cost_product = pd.read_sql_query(highest_cost_query, engine)

    # Displaying the result in Streamlit
    st.header("Top 5 Highest Average Cost Products")
    st.dataframe(highest_cost_product)

    # Plotting
    colormap = plt.cm.plasma
    colors = colormap(np.linspace(0, 1, len(highest_cost_product)))

    plt.figure(figsize=(14, 7))
    bars = plt.bar(highest_cost_product['Product_Name'], highest_cost_product['Average_Cost'], color=colors)

    plt.xlabel('Product Name', fontsize=12)
    plt.ylabel('Average Cost', fontsize=12)
    plt.title('Top 5 Highest Average Cost Products', fontsize=16)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')

    plt.xticks(rotation=45, ha='right')

    # Display the plot in Streamlit
    st.header("Top 5 Highest Average Cost Products - Average Cost Chart")
    st.pyplot(plt)




    # Eklenen kod başlangıcı
    category_counts = product_df['Product_Category'].value_counts()

    # Generate a color palette using the 'hsv' colormap
    colormap = plt.cm.hsv
    colors = colormap(np.linspace(0, 1, len(category_counts)))

    # Create the pie chart
    plt.figure(figsize=(14, 7))
    category_counts.plot.pie(autopct='%1.0f%%', colors=colors, textprops={'size': 14})
    plt.title('Product Category Distribution', fontsize=16)
    
    # Display the plot in Streamlit
    st.header("Product Category Distribution")
    st.pyplot(plt)



    # Eklenen kod başlangıcı
    location_counts = store_df['Store_Location'].value_counts()

    # Generate a color palette using the 'hsv' colormap
    colormap = plt.cm.hsv
    colors = colormap(np.linspace(0, 1, len(location_counts)))

    # Create the pie chart
    plt.figure(figsize=(14, 7))
    location_counts.plot.pie(autopct='%1.0f%%', colors=colors, textprops={'size': 14})
    plt.title('Store Location Distribution', fontsize=16)
    
    # Display the plot in Streamlit
    st.header("Store Location Distribution")
    st.pyplot(plt)


    # Eklenen kod başlangıcı
    aggPrice = {'Total_Sale': ['sum', 'mean', 'count']}
    aggCost = {'Total_Cost': ['sum', 'mean', 'count']}
    price_category = sales4.groupby(['Store_Location']).agg(aggPrice)
    cost_category = sales4.groupby(['Store_Location']).agg(aggCost)

    # Display the result in Streamlit
    st.header("Total Sales Statistics by Store Location")
    st.write(price_category)

    st.header("Total Cost Statistics by Store Location")
    st.write(cost_category)



    # Eklenen kod başlangıcı
    sales4['Date'] = pd.to_datetime(sales4['Date'], format='%Y-%m-%d')

    # Extract the year from the 'Date' column
    sales4['Date_Year'] = sales4['Date'].dt.year

    # Group by 'Date_Year', 'Product_Category', and 'Store_Location' and count the 'Total_Sale'
    category_location_sales = sales4.groupby(['Date_Year', 'Product_Category', 'Store_Location'])['Total_Sale'].count().sort_values(ascending=False).reset_index()

    # Display the result in Streamlit
    st.header("Sales Count by Year, Product Category, and Store Location")
    st.write(category_location_sales)



    # Yeni eklenen kod başlangıcı
    downtown = category_location_sales[category_location_sales['Store_Location'] == 'Downtown']
    comercial = category_location_sales[category_location_sales['Store_Location'] == 'Commercial']
    residencial = category_location_sales[category_location_sales['Store_Location'] == 'Residential']
    airport = category_location_sales[category_location_sales['Store_Location'] == 'Airport']
    data = [downtown, comercial, residencial, airport]

    plt.figure(figsize=(14, 20))
    for i, col in enumerate(data):
        axes = plt.subplot(4, 2, i + 1)
        sns.barplot(x='Product_Category', y='Total_Sale', data=col, errorbar=None, palette='viridis')
        plt.title(f"Product category {col['Store_Location'].unique()}", fontsize=14, fontweight='bold')
        plt.xticks(rotation=30, fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel('SALES')

        for cont in axes.containers:
            axes.bar_label(cont, fontsize=14)
    plt.tight_layout()

    # Display the plot in Streamlit
    st.header("Product Category Sales by Store Location")
    st.pyplot(plt)


    # Yeni eklenen kod başlangıcı
    total_sales_per_year_location = sales4.groupby(['Date_Year', 'Store_Location'])['Total_Sale'].count().reset_index()

    plt.figure(figsize=(10, 20))

    locations = total_sales_per_year_location['Store_Location'].unique()

    for i, location in enumerate(locations):
        location_data = total_sales_per_year_location[total_sales_per_year_location['Store_Location'] == location]

        ax = plt.subplot(4, 1, i + 1)

        colors = plt.cm.hsv(np.linspace(0, 1, len(location_data['Date_Year'].unique())))

        sns.barplot(x='Date_Year', y='Total_Sale', data=location_data, palette=colors, ax=ax)

        plt.title(f"Total Sales per Year - {location}", fontsize=14, fontweight='bold')

        plt.xticks(rotation=30, fontsize=14)
        plt.yticks(fontsize=14)

        plt.ylabel('Total Sales')

        for container in ax.containers:
            ax.bar_label(container, fontsize=14)

    plt.tight_layout()

    # Display the plot in Streamlit
    st.header("Total Sales per Year by Store Location")
    st.pyplot(plt)


    # Yeni eklenen kod başlangıcı
    category_location_sales2 = sales4.groupby(['Product_Name', 'Store_Location'])['Total_Sale'].mean().reset_index()
    downtown = category_location_sales2[category_location_sales2['Store_Location'] == 'Downtown'].sort_values(by='Total_Sale', ascending=False)
    comercial = category_location_sales2[category_location_sales2['Store_Location'] == 'Commercial'].sort_values(by='Total_Sale', ascending=False)
    residencial = category_location_sales2[category_location_sales2['Store_Location'] == 'Residential'].sort_values(by='Total_Sale', ascending=False)
    airport = category_location_sales2[category_location_sales2['Store_Location'] == 'Airport'].sort_values(by='Total_Sale', ascending=False)

    data = [downtown, comercial, residencial, airport]

    colormap = plt.cm.hsv  

    plt.figure(figsize=(16,35))

    for i, col in enumerate(data):
        axes = plt.subplot(4, 1, i + 1)
        
        num_colors = len(col['Product_Name'].unique())
        colors = colormap(np.linspace(0, 1, num_colors))

        sns.barplot(y=col['Product_Name'], x=col['Total_Sale'], ci=None, palette=colors)
        plt.title(f"Average Sales per Product in {col['Store_Location'].unique()[0]} ($)", fontsize=14, fontweight='bold')
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.xlabel('SALES($)')

        for cont in axes.containers:
            axes.bar_label(cont, fontsize=14)

    plt.tight_layout()

    # Display the plot in Streamlit
    st.header("Average Sales per Product by Store Location ($)")
    st.pyplot(plt)

if __name__ == "__main__":
    main()
