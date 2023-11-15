
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from matplotlib import colors
import warnings
warnings.filterwarnings("ignore")
import ast
from collections import Counter
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.express as px
from IPython import display

    
def read_dataset():
    '''for dirname, _, filenames in os.walk('/usr/PP'):
        for filename in filenames:
            print(os.path.join(dirname, filename))'''
    df = pd.read_csv("jobs.csv", encoding="utf-16").drop(['ID'], axis=1).drop_duplicates() 
    #Reading the csv file and dropping the duplicates while reading itself for further processing and cleaning!
    print("The dimensions of the imported dataset::",df.shape)
    return df
    
def cleanse_dataset(df):
    #The dataset has too many NULL values. Dropping those especially from the very important attributes - Low, Mean and High Salaries, 
    #since these are of utmost importance for data professionals like us!!
    df=df.dropna(subset=['Low_Salary','High_Salary','Mean_Salary'])
    #Mean Salary is very much to be taken care of and we, the data professionals cannot accept salaries in negative!
    #Hence, removing the least significant mean_salary values including the negatives.
    main_label = 'Mean_Salary'
    df[main_label] = df[main_label]*1e-3
    df = df[df[main_label]>0]
    df[main_label] = df[main_label]*1e+3 #Setting back to original decimal places
    #Concatenating city and state
    df['City'] = (df['City'] + ', ' + df['State'])
    #Filling nan values
    df['Profile'] = df['Profile'].fillna('None')
    df['City'] = df['City'].fillna('Remote')
    df['Remote'] = df['Remote'].fillna('On-Site')
    #The dataset seems to be a bit in spanish, so converting the calendrical spanish values to English...!
    df['Frecuency_Salary'] = df['Frecuency_Salary'].replace(['año','hora','mes', 'día', 'semana'],['annum','hour','month','day','week'])
    
    return df
    
def skills_extract(df):
    #The Skills column seems a list of lists...A very dirty dataset indeed!
    #Hereby extracting every skill value and appending to a separate list 'l2' that can be further processed.
    l1=list()
    for i in range (0,len(list(df['Skills']))):
        l1.append((list(df['Skills'])[i][2:-2].replace('\'','').split(',')))    
    l2=list()
    for i in range(len(l1)):
        for j in range(len(l1[i])):
            l2.append(l1[i][j])
    
    return l2

def visualize_skills_required(l2):
    words1=list()
    for i in l2:
        words1.append(i.lstrip())
    
    #Count occurrences of each word
    word_counts = Counter(words1)
    color_blind_palette = ["#DC267F", "#785EF0", "#648FFF", "#FE6100", "#FFB000", "#E80E8D", "#2E8766",
                        "#95ABAC", "#65B925", "#906A42", "#C4D537", "#344A52", "#6C6E06", "#1C4DD2", "#216E00",
                        "#2E03E5", "#A94424", "#7F6EA9", "#9B8453", "#380721"]
    # Extract words and their counts
    words = list(word_counts.keys())
    counts = list(word_counts.values())
    indexes = np.arange(len(words))
    
    #Plot the bar graph
    plt.figure(figsize=(27,6))
    ms_skills = sns.barplot(x=words,y=np.log2(counts), palette=color_blind_palette)

    plt.rcParams['font.size'] = '9'
    width = 0.7
    plt.tick_params(axis="x", rotation=90)
    plt.xlabel('Words',fontsize=0.02)
    plt.ylabel('Occurrences in log2 scale',fontsize=10)
    plt.title('Word Occurrences in the List of Skills')
    for patch in ms_skills.patches:
                ms_skills.annotate("%.0f" % patch.get_height(), (patch.get_x() + patch.get_width() / 2., patch.get_height()),
                    ha='center', va='center', fontsize=8, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.show()
    print("So, the most required skills for a data professional as can be seen from the graph seem to be--")
    
def Top_20(df):
    
    dfp = df['Jobs_Group'].value_counts().head(20).sort_values(ascending = True).reset_index()
    dfl = df['City'].value_counts().head(20).sort_values(ascending = True).reset_index()
    dfc = df['Company'].value_counts().head(20).sort_values(ascending = True).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(x = dfp['count'],
                        y = dfp['Jobs_Group'],
                        orientation='h',
                        name = 'Job_Position',
                        marker = dict(color = 'Green')))

    fig.add_trace(go.Bar(x = dfl['count'],
                        y = dfl['City'],
                        orientation='h',
                        name = 'Location',
                        marker = dict(color = 'SeaGreen')))

    fig.add_trace(go.Bar(x = dfc['count'],
                        y = dfc['Company'],
                        orientation='h',
                        name = 'Company',
                        marker = dict(color = 'darkgreen')))

    fig.update_layout(
        updatemenus=[
            dict(
                type = "buttons",
                direction="left",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.16,
                xanchor="left",
                y=1.12,
                yanchor="top",
                font = dict(color = 'Indigo',size = 12),
                buttons=list([
                    dict(label="All",
                        method="update",
                        args=[ {"visible": [True, True, True]},
                                {'showlegend' : True} , {"font":10}
                            ]),
                    dict(label="Job_Position",
                        method="update",
                        args=[ {"visible": [True, False, False]},
                                {'showlegend' : True}
                            ]),
                    dict(label='Location',
                        method="update",
                        args=[ {"visible": [False, True, False]},
                                {'showlegend' : True}
                        ]),
                    dict(label='Company',
                        method="update",
                        args=[ {"visible": [False, False, True]},
                                {'showlegend' : True}]),
                ]),
            )])
    fig.update_layout(
        annotations=[
            dict(text="Choose:", showarrow=False,
                x=0, y=1.075, yref="paper", align="right",
                font=dict(size=12,color = 'Black'))])

    fig.update_layout(title ="Top 20 Job_Positions, Locations and Companies",
                    title_x = 0.5,
                    title_font = dict(size = 20, color = 'MidnightBlue'))
    fig.show()
    print(".....Welcome to one of the interactive graphs of our analytics.....")
    print("Financial Analyst, Business Analyst, Data Analyst being the TOP 3 job roles, woahhh!!")    
    print("Citi hires the most!!!")
    print("And just look at the location---it's Remote at which the most number of data scientists work from!! Who doesn't love working remotely, coz I do:)")

def job_level_distribution(df):
    
    dfd1 = df[df['Profile']== 'Junior']
    dfd2 = df[df['Profile']== 'Senior']
    dfd3 = df[df['Profile']== 'Lead']

    redf1 = dfd1["Jobs_Group"].value_counts()[:15].reset_index()
    redf2 = dfd2["Jobs_Group"].value_counts()[:15].reset_index()
    redf3 = dfd3["Jobs_Group"].value_counts()[:15].reset_index()

    fig = go.Figure()

    fig.add_trace(go.Bar(x = redf1["Jobs_Group"],
                        y = (redf1["count"]),
                        marker = dict(color = 'lightcoral'),
                        name = 'Junior'))

    fig.add_trace(go.Bar(x = redf2['Jobs_Group'],
                        y = (redf2["count"]),
                        name = 'Senior',
                        marker = dict(color = 'lightblue')))

    fig.add_trace(go.Bar(x = redf3['Jobs_Group'],
                        y = (redf3["count"]),
                        name = 'Lead',
                        marker = dict(color = 'pink')))


    fig.update_layout(
        updatemenus=[
            dict(
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.13,
                xanchor="left",
                y=1.12,
                yanchor="top",
                font = dict(color = 'Indigo',size = 14),
                buttons=list([
                    dict(label="All",
                        method="update",
                        args=[ {"visible": [True, True, True]},
                                {'showlegend' : True}
                            ]),
                    dict(label="Junior",
                        method="update",
                        args=[ {"visible": [True, False, False]},
                                {'showlegend' : True}
                            ]),
                    dict(label='Senior',
                        method="update",
                        args=[ {"visible": [False, True, False]},
                                {'showlegend' : True}
                        ]),
                    dict(label='Lead',
                        method="update",
                        args=[ {"visible": [False, False, True]},
                                {'showlegend' : True}
                            ]),

                ]),
            )])

    fig.update_layout(
        annotations=[
            dict(text="Choose:", showarrow=False,
                x=0, y=1.075, yref="paper", align="right",
                font=dict(size=14,color = 'DarkSlateBlue'))])

    fig.update_layout(title ="The distribution of Jobs Groups by 3 Profiles",
                    title_x = 0.5,
                    title_font = dict(size = 15, color = 'MidnightBlue'))

    print("---Interactive Plot---")
    fig.show()
    print("As the corporate trend follows, here also we see that - more the experience, more you get the chance to explore!!")
    print("Leads and Seniors are more in number! :)")
    
def jobrole_vs_salary(df):
    
    #Here, we consider the mean salaries for various job roles from our dataset..
    import matplotlib.pyplot as plt

    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Jobs_Group'], df['Mean_Salary'],label='Mean Salary',color = 'red',marker = '*')

    # Adding labels and title
    plt.xlabel('Job Profiles')
    plt.ylabel(' in USD per Annum')
    plt.title('Salaries vs Job Profiles')
    plt.legend()  # Show legend

    # Rotating x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Show the plot
    plt.tight_layout()
    plt.show()
    print("CFO's, Data Scientists, Data Engineers, ML Engineers earn quite well!!!!")
    
    print("----------------------Salary Level Comparison---------------------------")
    print("MIN, MEAN AND MAX SALARIES VS Various Job Profiles...")
    fig=make_subplots(rows=3,cols=1,subplot_titles=('<i>Minimum Salaries', '<i>Mean Salaries', '<i>Maximum Salaries'))
    fig.add_trace(go.Scatter(x=df['Jobs_Group'],y=df['Low_Salary'],name='Minimum',mode='markers'),row=1,col=1)
    fig.add_trace(go.Scatter(x=df['Jobs_Group'],y=df['Mean_Salary'],name='Mean',mode='markers'),row=2,col=1)
    fig.add_trace(go.Scatter(x=df['Jobs_Group'],y=df['High_Salary'],name='Maximum',mode='markers'),row=3,col=1)

    fig.update_layout(height=1000, width=800, title_text='<b>Salary Comparison', font_size=12)
    fig.update_layout(template='plotly_dark', title_x=0.5, font_family='Courier New', showlegend=True)
    fig.show()
def ThreeD_Plot(df):
    
    fig = px.scatter_3d(df, x='Profile', y='Remote', z='Mean_Salary',
                    color='Mean_Salary')
    fig.update_layout(title='<b> 3D Scatter Plot <b>', title_x=0.5,
                    titlefont=dict({'size':28, 'family': 'Courier New', 'color':'white'}),
                    template='plotly_dark',
                    width=900, height=500,
                    )
    fig.update_layout(scene = dict(
                        xaxis = dict(
                            backgroundcolor="rgb(200, 200, 230)",
                            gridcolor="white",
                            showbackground=True,
                            zerolinecolor="black",),
                        yaxis = dict(
                            backgroundcolor="rgb(230, 200,230)",
                            gridcolor="white",
                            showbackground=True,
                            zerolinecolor="black"),
                        zaxis = dict(
                            backgroundcolor="rgb(230, 230,200)",
                            gridcolor="white",
                            showbackground=True,
                            zerolinecolor="black"),
                                ),
                    )
    fig.show()
    print("This is the 3-Dimensional Plot created using Plotly!!")
    print("You can hover your cursor to check what salary can be expected at kinda location and for which job level. :)")
    print("Remote location plus a senior level is expected to live a luxurious life!!")
    
def test_graph(df):
    df1 = pd.DataFrame(df,columns=['Company','High_Salary','Low_Salary'])
    cd = df1.groupby('Company')[['High_Salary','Low_Salary']].max().reset_index()
    fig,ax=plt.subplots()
    fig.figsize=(6, 4)
    #Plot Scatter plot
    plt.scatter((cd['High_Salary'].clip(lower=1)),(cd['Low_Salary'].clip(lower=1)),c='purple',marker='o')
    plt.xlabel('(Max Salary)')
    plt.ylabel('(Min Salary)')
    plt.title('Minimum Salary vs. Maximum Salary')
    plt.grid(True)
    plt.show()
    
    #Violin Plots---
    fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
    ax1.violinplot(df['Low_Salary'],showmedians=True,showmeans=True,vert=True)
    ax2.violinplot(df['High_Salary'],showmedians=True,showmeans=True,vert=True)
    
    print("This graph is just a test and we passed this test as the maximum salary is always higher than the minimum salary!!")
    
    
def avg_sal_by_loc(df):
    exp_salary = df.groupby('Remote')['High_Salary'].mean()
    plt.figure(figsize = (8,6))
    ax = sns.barplot(x = exp_salary.index, y = exp_salary.values, palette = 'Reds')
    plt.title('Average Salary by Location Level', fontsize=12, fontweight='bold')
    plt.xlabel('Location', fontsize=12, fontweight='bold')
    plt.ylabel('Average Salary (USD)', fontsize=12, fontweight='bold')

    for container in ax.containers:
        ax.bar_label(container,
                    padding = -50,
                    fontsize = 12,
                    bbox = {'boxstyle': 'circle', 'edgecolor': 'black', 'facecolor': 'lightblue'},
                    label_type="edge",
                    fontweight = 'bold'
                    )

    # Customize the background color
    ax.set_facecolor("#f4f4f4")
    # Remove the grid lines
    ax.grid(False)

    plt.show()
    print("Companies prefer hybrid mode of work to get you paid more on average...")
    
def Pie(df):
    
    fig=px.pie(df, values=df['Remote'].value_counts().values,
           names=df['Remote'].value_counts().index, title='Percentage of remote ratio')
    fig.show()
    print("Companies call you for On-Site work more as it's likely that projects are distributed around the globe...")
    

# Function to perform linear search and visualize the relationship between companies and salaries using a line graph
def linear_search_and_visualize_line_chart(df,company_name):
    
    print("-------------------This Function is the centre of interest for the entire Analytics-----------------------")
    df = df.head(30)
    lc = list(df['Company'])
    ls = list(df['High_Salary'])
    fig, ax = plt.subplots()

    ax.plot(lc, ls, marker='o', linestyle='-', color='lightblue')
    ax.set_xticks(lc)
    ax.set_xticklabels(lc, rotation=35, ha='right',fontsize=7)
    ax.set_ylabel('Salary')
    ax.set_title('Company vs Salary Visualization - Linear Search')
    ax.grid(True)
    found = False

    for i, name in enumerate(lc):
        if name == company_name:
            ax.plot(name, ls[i], marker='o', markersize=8, color='red', label='Found')
            ax.text(name, ls[i] + 2000, f'{ls[i]}', ha='center', va='bottom', color='black')
            ax.vlines(name, 0, ls[i], linestyles='dashed', color='orange', alpha=0.5)
            found = True
        else:
            ax.plot(name, ls[i], marker='o', linestyle='', color='blue')
            
        #this display.display gives the visual effect/animation to the plot
        #display.display(fig)
        #display.clear_output(wait=True)
        plt.show(block=False)
        plt.pause(0.01)
        #plt.savefig(f'frames/frame_{i:03d}.png')
        #plt.close()

    if not found:
        print(f"{company_name} not found in the DataFrame.")

    plt.show()


# Function to perform binary search and visualize the relationship between companies and salaries using a line graph
def binary_search_and_visualize_line_chart(df,company_name):
    
    print("-------------------This Function is the centre of interest for the entire Analytics-----------------------")
    df = df.head(30)
    
    lc = list(map(str, df['Company'])) 
    ls = list(df['High_Salary'])

    # Sort both lists based on the 'Company'
    sorted_data = sorted(zip(lc, ls), key=lambda x: x[0])
    lc, ls = zip(*sorted_data)
    fig, ax = plt.subplots()

    ax.plot(lc, ls, marker='o', linestyle='-', color='tomato', label='All Companies')
    ax.set_xticks(lc)
    ax.set_xticklabels(lc, rotation=35, ha='right',fontsize=7)
    ax.set_ylabel('Salary in USD')
    ax.set_title('Company vs Salary Visualization - Binary Search')
    ax.grid(True)

    found = False

    # Perform binary search on the sorted data
    left, right = 0, len(lc) - 1
    sequence_number = 1  # Initialize sequence number

    while left <= right:
        mid = (left + right) // 2

        # Highlight the current dot
        ax.plot(lc[mid], ls[mid], marker='o', markersize=10, color='green', label='Current Search')

        # Print the sequence number on top of the highlighted dot
        ax.text(lc[mid], ls[mid] + 2000, str(sequence_number), ha='center', va='top', color='white', fontsize=8)

        if lc[mid] == company_name:
            # Highlight the final result with a different color
            ax.plot(lc[mid], ls[mid], marker='D', markersize=10, color='black', label='Final Result')
            ax.text(lc[mid], ls[mid] + 2000, f'{ls[mid]}', ha='center', va='bottom', color='black')
            ax.vlines(lc[mid], 0, ls[mid], linestyles='dashed', color='darkgreen', alpha=0.5)
            found = True
            break
        elif lc[mid] < company_name:
            left = mid + 1
        else:
            right = mid - 1

        # Increment the sequence number for the next iteration
        sequence_number += 1

        # Update the plot
        plt.show(block=False)
        plt.pause(1.5)
        #display.display(fig)
        #display.clear_output(wait=True)

    if not found:
        print(f"{company_name} not found in the DataFrame.")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    #import_lib()
    df = read_dataset()
    df = cleanse_dataset(df)
    print("Welcome to the Data Science Job Market in US - Analysis")
    
    print("---------------------------------------------------------------------------------------------------------")
    print("Let's visualize and analyse the data!!!")
    print(df.head(10))
    ans = input("Press one of the options' number code to get you visualize different analytical techniques of our dataset...\n1. What all skills are mostly required for a data professional? \n2. View the Top 20 Job_Positions, Locations and Companies \n3. Job Level Distribution analytics \n4. What job role will pay you what? \n5. Wanna view 3D - Plot?  \n6. Test whether our analysis is going correct \n7. Average Salaries by Location \n8. PIE Chart \n9. Applying DSA Concepts to our analysis and visualising realtime results \n10. Exit(by pressing -1)")    
    #visualize_skills_required(l2)
    print(ans)
    
    while(ans!='-1'):
        ans = input("Press one of the options' number code to get you visualize different analytical techniques of our dataset...\n1. What all skills are mostly required for a data professional? \n2. View the Top 20 Job_Positions, Locations and Companies \n3. Job Level Distribution analytics \n4. What job role will pay you what? \n5. Wanna view 3D - Plot?  \n6. Test whether our analysis is going correct \n7. Average Salaries by Location \n8. PIE Chart \n9. Applying DSA Concepts(LINEAR SEARCH) to our analysis and visualising realtime results \n10. Applying DSA Concepts(BINARY SEARCH) to our analysis and visualising realtime results  \n11. Exit(by pressing -1)\n")    
        #visualize_skills_required(l2)
        if ans=='1':
            print("Let's get the skillset extracted from the very much dirty dataset!!!---Hang On!")
            l2 = skills_extract(df)
            visualize_skills_required(l2)
            continue
        elif ans=='2':
            Top_20(df)
            continue
        elif ans=='3':
            job_level_distribution(df)
            continue
        elif ans=='4':
            jobrole_vs_salary(df)
            continue
        elif ans=='5':
            ThreeD_Plot(df)
            continue
        elif ans=='6':
            test_graph(df)
            continue
        elif ans=='7':
            avg_sal_by_loc(df)
            continue
        elif ans=='8':
            Pie(df)
            continue
        elif ans=='9':
            #if not os.path.exists('frames'):
            #    os.makedirs('frames')
            print("This option will lead you through the linear search visually")
            print("Which company are you interested in? Wanna see the highest salary that it pays?")
            company_name = input("Enter the company name")
            linear_search_and_visualize_line_chart(df,company_name)
            continue
        elif ans=='10':
            print("This option will lead you through the binary search visually")
            print("Which company are you interested in? Wanna see the highest salary that it pays?")
            company_name = input("Enter the company name::")
            binary_search_and_visualize_line_chart(df,company_name)
            continue
        elif ans=='-1':
            print("THANK YOU!")
            break
        else:
            print("Please enter valid input")
            continue
        