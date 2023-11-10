import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset into a pandas DataFrame
df = pd.read_csv('C:/Users/BANTA/Desktop/ML projects/dataSets/zomato.csv')

# # Display the first few rows of the DataFrame
# print(df.head())

# # Check the dimensions of the DataFrame
# print("Shape of the DataFrame:", df.shape)

# # Check the column names
# print("Column names:", df.columns)

# # Check the data types of each column
# print("Data types:\n", df.dtypes)

# # Check for missing values
# print("Missing values:\n", df.isnull().sum())



# Location of the restaurants
plt.figure(figsize=(12, 6))
location_counts = df['location'].value_counts().head(10)
sns.barplot(x=location_counts, y=location_counts.index)
plt.title('Top 10 Restaurant Locations')
plt.xlabel('Count')
plt.ylabel('Location')
plt.show()

# Approximate price of food
plt.figure(figsize=(10, 6))
sns.histplot(df['approx_cost(for two people)'], bins=10)
plt.title('Cost Distribution')
plt.xlabel('Approximate Cost (for two people)')
plt.ylabel('Count')
plt.xticks(rotation='vertical')
plt.show()

# Famous neighborhood kind of food
plt.figure(figsize=(12, 6))
cuisine_counts = df['cuisines'].value_counts().head(10)
sns.barplot(x=cuisine_counts, y=cuisine_counts.index)
plt.title('Top 10 Popular Cuisines')
plt.xlabel('Count')
plt.ylabel('Cuisine')
plt.show()

# Theme-based restaurant or not
plt.figure(figsize=(8, 6))
theme_counts = df['listed_in(type)'].value_counts()
sns.countplot(x='listed_in(type)', data=df)
plt.title('Theme-based Restaurants')
plt.xlabel('Type')
plt.ylabel('Count')
plt.xticks(rotation='vertical')
plt.show()

# Locality with the highest number of restaurants
plt.figure(figsize=(12, 6))
locality_counts = df['listed_in(city)'].value_counts().head(10)
sns.barplot(x=locality_counts, y=locality_counts.index)
plt.title('Top 10 Localities with Most Restaurants')
plt.xlabel('Count')
plt.ylabel('Locality')
plt.show()

# People's needs
plt.figure(figsize=(8, 6))
sns.countplot(x='online_order', data=df)
plt.title('Online Ordering')
plt.xlabel('Online Order')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='book_table', data=df)
plt.title('Table Booking')
plt.xlabel('Table Booking')
plt.ylabel('Count')
plt.show()

# # Rating vs. Votes
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='rate', y='votes', data=df)
# plt.title('Rating vs. Votes')
# plt.xlabel('Rating')
# plt.ylabel('Votes')
# plt.show()

# Restaurant Type by Location
# plt.figure(figsize=(12, 6))
# cross_tab = pd.crosstab(df['location'], df['rest_type'])
# sns.heatmap(cross_tab, cmap='YlGnBu')
# plt.title('Restaurant Type by Location')
# plt.xlabel('Restaurant Type')
# plt.ylabel('Location')
# plt.show()

# # Cost Distribution by Cuisine
# plt.figure(figsize=(12, 6))
# sns.violinplot(x='approx_cost(for two people)', y='cuisines', data=df)
# plt.title('Cost Distribution by Cuisine')
# plt.xlabel('Approximate Cost (for two people)')
# plt.ylabel('Cuisine')
# plt.show()

# # Rating Distribution by Location
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='location', y='rate', data=df)
# plt.title('Rating Distribution by Location')
# plt.xlabel('Location')
# plt.ylabel('Rating')
# plt.xticks(rotation=90)
# plt.show()

# Cuisine Distribution by Location
plt.figure(figsize=(12, 6))
cuisine_counts = df.groupby('location')['cuisines'].nunique().sort_values(ascending=False).head(10)
sns.barplot(x=cuisine_counts.index, y=cuisine_counts.values)
plt.title('Cuisine Distribution by Location')
plt.xlabel('Location')
plt.ylabel('Number of Cuisines')
plt.xticks(rotation=90)
plt.show()

# Restaurant Type Distribution by Location
plt.figure(figsize=(12, 6))
restaurant_type_counts = df.groupby('location')['rest_type'].nunique().sort_values(ascending=False).head(10)
sns.barplot(x=restaurant_type_counts.index, y=restaurant_type_counts.values)
plt.title('Restaurant Type Distribution by Location')
plt.xlabel('Location')
plt.ylabel('Number of Restaurant Types')
plt.xticks(rotation=90)
plt.show()

# Rating Distribution by Restaurant Type
plt.figure(figsize=(12, 6))
sns.boxplot(x='rest_type', y='rate', data=df)
plt.title('Rating Distribution by Restaurant Type')
plt.xlabel('Restaurant Type')
plt.ylabel('Rating')
plt.xticks(rotation=90)
plt.show()

# Online Ordering and Table Booking
plt.figure(figsize=(10, 6))
sns.countplot(x='online_order', data=df, hue='book_table')
plt.title('Online Ordering and Table Booking')
plt.xlabel('Online Order')
plt.ylabel('Count')
plt.show()

# # Rating Distribution by Cost
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='approx_cost(for two people)', y='rate', data=df)
# plt.title('Rating Distribution by Cost')
# plt.xlabel('Approximate Cost (for two people)')
# plt.ylabel('Rating')
# plt.xticks(rotation=90)
# plt.show()

# # Rating Distribution by Delivery Option
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='listed_in(type)', y='rate', hue='listed_in(city)', data=df)
# plt.title('Rating Distribution by Delivery Option')
# plt.xlabel('Delivery Option')
# plt.ylabel('Rating')
# plt.xticks(rotation=90)
# plt.legend(title='City')
# plt.show()

# Restaurant Chains
plt.figure(figsize=(12, 6))
restaurant_chain_counts = df['name'].value_counts().head(10)
sns.barplot(x=restaurant_chain_counts.index, y=restaurant_chain_counts.values)
plt.title('Top 10 Restaurant Chains')
plt.xlabel('Restaurant Chain')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

# # Rating vs. Online Ordering/Book Table
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='rate', y='online_order', hue='book_table', data=df)
# plt.title('Rating vs. Online Ordering/Book Table')
# plt.xlabel('Rating')
# plt.ylabel('Online Order')
# plt.legend(title='Book Table')
# plt.show()

# Reviews Sentiment Analysis
# Perform sentiment analysis on customer reviews and visualize the sentiment distribution

# Menu Item Popularity
plt.figure(figsize=(12, 6))
menu_items_counts = df['menu_item'].value_counts().head(10)
sns.barplot(x=menu_items_counts, y=menu_items_counts.index)
plt.title('Top 10 Popular Menu Items')
plt.xlabel('Count')
plt.ylabel('Menu Item')
plt.show()
'''
information gotten from the visualization 

1. Location of the restaurants:
   - The visualization provides the top 10 restaurant locations based on the count of restaurants in each location.
   - It helps identify the areas with the highest concentration of restaurants.

2. Approximate price of food:
   - The histogram shows the distribution of approximate costs for two people.
   - It helps understand the price range of restaurants in the dataset.

3. Famous neighborhood kind of food:
   - The bar chart displays the top 10 popular cuisines based on the count of restaurants serving each cuisine.
   - It provides insights into the most popular cuisines in the dataset.

4. Theme-based restaurant or not:
   - The countplot shows the number of theme-based restaurants compared to non-theme-based restaurants.
   - It helps understand the prevalence of theme-based restaurants in the dataset.

5. Locality with the highest number of restaurants:
   - The bar chart displays the top 10 localities with the most restaurants based on the count of restaurants in each locality.
   - It helps identify the areas with the highest restaurant density.

6. People's needs:
   - The countplot shows the count of restaurants offering online ordering.
   - It provides insights into the popularity of online ordering among restaurants.

7. Rating vs. Votes:
   - The scatter plot shows the relationship between ratings and the number of votes received by restaurants.
   - It helps analyze the correlation between ratings and popularity.

8. Restaurant Type by Location:
   - The heatmap shows the distribution of restaurant types across different locations.
   - It helps identify the dominant restaurant types in each area.

9. Cost Distribution by Cuisine:
   - The violin plot displays the distribution of approximate costs for two people for different cuisines.
   - It helps compare the cost distribution across various cuisines.

10. Rating vs. Online Ordering/Book Table:
    - The boxplot shows the relationship between ratings and the availability of online ordering and table booking options.
    - It helps analyze the impact of these features on ratings.

11. Menu Item Popularity:
    - The bar chart displays the top 10 popular menu items based on the count of mentions in the dataset.
    - It provides insights into the most loved or frequently ordered menu items.

These visualizations offer valuable insights into the Zomato dataset, allowing me to understand various aspects such as location distribution, pricing, cuisine popularity, restaurant types, customer preferences, and more.'''


