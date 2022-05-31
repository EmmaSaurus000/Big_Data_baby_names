"""
Author: Emma Saurus
Date: 21/5/2022
Project: Big Data Assignment 5: Baby names
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import axes
import matplotlib.patches as mpatches
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

plt.style.use('dark_background')

names_df = None


""" Function reads txt files and saves compiled dataframe as feather binary"""
def load_name_data():
    global names_df
    try: 
        for yr in range(1880, 2019):
            year_string = str(yr)
            this_year = './names/yob' + year_string + '.txt'
            this_df = pd.read_csv(this_year, names=['Name', 'Sex', 'Count'], dtype={'Year': int, 'Name': str, 'Sex': str, 'Count': int})
            # this_df['Year'] = str(yr)
            this_df.insert(0, 'Year', yr)

            if names_df is None:
                names_df = this_df
                print('In if.', end=' ')
            else:
                names_df = pd.concat([names_df, this_df], ignore_index=True)
                print('In else.', end=' ')

            print('Saved :', end=' ')
            print(yr)
    except Exception as e:
        print('Failed to load names_df')
        print(e)
    
    try:
        names_df.to_feather('./outputs/names.feather')
    except Exception as e:
        print('Failed to save as feather')
        print(e)


""" Function reads feather binary into dataframe """
def load_feather():
    global names_df
    names_df = pd.read_feather('./outputs/names.feather', columns=['Year', 'Name', 'Sex', 'Count'])
    names_df['Name'] = names_df['Name'].astype('string')
    names_df['Sex'] = names_df['Sex'].astype('string')
    # print(names_df)
    # print(names_df.dtypes)


""" Function adds value labels for plots """
def add_labels(Xs, Ys):
    for i in range(1, len(Xs) + 1):
        plt.text(i, Ys[i-1], Ys[i-1])


""" Return scatter plot of names given to most babies within a year """
def task_1_a():
    global names_df
    
    most_pop_per_year = names_df.sort_values(by='Count', ascending=False, inplace=False)
    most_pop_per_year = most_pop_per_year.head(20)
    print('\nMost babies given same name in a year, 1880-2015')
    print(most_pop_per_year)

    # Plot
    colors = []
    try:
        for s in most_pop_per_year['Sex']:
            if s == 'F':
                colors.append('violet')
            else:
                colors.append('cornflowerblue')
    except Exception as e:
        print('Failed to plot 1_a')
        print(e)

    try:
        try:
            annotations=most_pop_per_year['Year'].head(7)
            annotations = annotations.array
            annotations.insert(0,0)
            print(annotations)
            annotations_x =most_pop_per_year['Name'].head(7)
            annotations_x = annotations_x.array
            annotations_x.insert(0,'Linda')
            annotations_y =most_pop_per_year['Count'].head(7)
            annotations_y = annotations_y.array
            annotations_y.insert(0,0)
            print(annotations_x)
        except Exception as e:
            print('Failed stage 0')
            print(e)

        try:
            fig, ax = plt.subplots()
            #ax.bar(np.arange(len(most_pop_per_year['Name'])), most_pop_per_year['Count'], label='Number of babies with name', color=colors)
            ax.scatter(most_pop_per_year['Name'], most_pop_per_year['Count'], label='Number of babies with name', color=colors)
            ax.set_xticks(most_pop_per_year['Name'])
            # ax.set_xticklabels(most_pop_per_year['Name'], rotation=90, fontsize=8)
            plt.setp(ax.get_xticklabels(), rotation=90)
        except Exception as e:
            print('Failed stage 1')
            print(e)
        
        try:
            ax.annotate('offset pixels', xy=(10,10), xytext=(5,5), textcoords='offset pixels')
        except Exception as e:
            (print('Failed to annotate - stage 1.5'))
            print(e)

        try:
            plt.subplots_adjust(left=0.15, bottom=0.2)
            ax.set_title('Most babies with same name in single year')
            ax.set_xlabel('Name')
            ax.set_ylabel('Number of babies')
            male_leg = mpatches.Patch(color='cornflowerblue', label='Male')
            female_leg = mpatches.Patch(color='violet', label='Female')
            ax.legend(handles=[male_leg, female_leg])
        except Exception as e:
            print('Failed stage 2')
            print(e)
        
        try:
            # ax.annotate('1947', xy=(0.1, 99689), xycoords='axes fraction', xytext=(0.1, 0.95), textcoords='offset pixels')
            ax.text('Linda', 99689, '1947')
            arr_length = len(annotations)
            # for i in enumerate(annotations):
            for i in range(0, arr_length):
                #plt.annotate(annotations[i], (annotations_x[i], annotations_y[i]))
                if i:
                    text = ax.text(annotations_x[i], annotations_y[i], annotations[i], color='w')
                    # text = ax.text(annotations_x[i], annotations_y[i], annotations[i], color='w', xytext=(5,5), textcoords='offset pixels')
                else: print('No i')
        
        except Exception as e:
            print('Failed stage 3')
            print(e)

        try:
            # plt.tight_layout
            plt.savefig('./outputs/Task_1_a.png')
            plt.show()
            plt.close()
        except Exception as e:
            print('Failed stage 4')
            print(e)
    except Exception as e:
        print('Failed to plot 1_a')
        print(e)


""" Return a bar graph of the top 25 names chosen most over the entire dataset period """
def task_1_b():
    global names_df

    all_time_df = names_df
    all_time_df['Sum'] = all_time_df.groupby('Name')['Count'].transform(sum)
    all_time_df['Sum_by_Sex'] = all_time_df.groupby(['Name', 'Sex'])['Count'].transform(sum)
    print('\n post group:')
    print(all_time_df.head(20))
    all_time_df = all_time_df.sort_values(by='Sum', ascending=False)
    all_time_df = all_time_df.drop_duplicates(subset=['Name', 'Sex'])
    all_time_df = all_time_df.head(50)
    all_time_df = all_time_df.drop(['Year', 'Count'], axis=1)
    print('\n Post sort, unique, drop:')
    print(all_time_df)

    female_df = all_time_df.loc[all_time_df['Sex'] == 'F']
    print('\nFemale :')
    print(female_df)
    male_df = all_time_df.loc[all_time_df['Sex'] == 'M']
    print('\nMale :')
    print(male_df)
    width = 0.35
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.15, bottom=0.3)

    ax.bar(male_df['Name'], male_df['Sum_by_Sex'], width, color='cornflowerblue')
    # ax.bar(female_df['Name'], female_df['Sum_by_Sex'], width, bottom=male_df['Sum'], color='violet')
    ax.bar(female_df['Name'], female_df['Sum_by_Sex'], width, color='violet')
    
    ax.set_title('Top 25 All-time most-popular names, US, 1880-2018', y=1.0, pad=20)
    ax.set_xlabel('Name', labelpad=20)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.set_ylabel('Number of babies (millions)', labelpad=20)
    plt.ylim(1200000,5500000)
    male_leg = mpatches.Patch(color='cornflowerblue', label='Male')
    female_leg = mpatches.Patch(color='violet', label='Female')
    ax.legend(handles=[male_leg, female_leg])

    plt.savefig('./outputs/Task_1_b.png')
    plt.show()
    plt.close()


""" Return a line plot of the number of babies registered per year """
def task_1_c():
    global names_df

    birth_rate = names_df
    birth_rate['Sum'] = birth_rate.groupby('Year')['Count'].transform(sum)
    birth_rate = birth_rate.drop_duplicates(subset=['Year'])
    print('\n Post unique:')
    print(birth_rate.head(50))

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.15, bottom=0.2)

    ax.plot(birth_rate['Year'], birth_rate['Sum'], color='red')
    
    ax.set_title('Babies registered per year, US, 1880-2018', y=1.0, pad=20)
    ax.set_xlabel('Year', labelpad=20)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.set_ylabel('Number of babies (millions)', labelpad=20)
    # plt.ylim(1200000,5500000)

    plt.savefig('./outputs/Task_1_c.png')
    plt.show()
    plt.close()


""" Return a heatmap of the top-10 most popular names for each year """
def task_2():
    global names_df

    all_time_df = names_df
    all_time_df = all_time_df.sort_values(by=['Year', 'Count'], ascending=[True,False])
    all_time_df = all_time_df.groupby('Year').head(10)

    gender_df = all_time_df.filter(['Name', 'Sex'], axis=1)
    gender_df = gender_df.sort_values(by=['Name'])
    print(gender_df)

    try:
        pivoted_df = pd.pivot_table(all_time_df, index="Name", columns="Year", values="Count", fill_value=0)
        #index_name = pasadena_df.index.name = 'fooyoo'
        #pasadena_df = pd.pivot_table(pasadena_df, index=["fooyoo", "DATE_YYYYMMDD"], columns="ELEMENT", values="DATA_VALUE")
        print('Pivot_tabled dataframe:')
        print(pivoted_df)
    except Exception as e:
        print('Pivot failed')
        print(e)

    colors = []
    try:
        for s in gender_df['Sex']:
            if s == 'F':
                colors.append('violet')
            else:
                colors.append('cornflowerblue')
    except Exception as e:
        print('Failed to plot 1_a')
        print(e)

    fig, ax = plt.subplots(figsize = (14, 7))
    plt.subplots_adjust(bottom=0.3)
    ax.scatter(all_time_df['Name'], all_time_df['Year'],s=5) #, color=colors)
    
    ax.set_title('Top-10 Baby Names per Year, US, 1880-2018', y=1.0, pad=20)
    ax.set_xlabel('Name', labelpad=20)
    plt.setp(ax.get_xticklabels(), rotation=90) 
    plt.setp(ax.get_yticklabels(), fontsize='10')
    ax.set_ylabel('Year', labelpad=20)

    plt.savefig('./outputs/Task_2.png')
    plt.show()
    plt.close()

    """
    pivoted_df.to_excel('./outputs/Top-10_names.xlsx')
    print(all_time_df.head(50))

    # plt.figure(figsize=(15, 15))
    plt.gcf().set_size_inches(20, 20)
    plt.scatter(all_time_df['Year'], all_time_df['Name'])
    plt.set_ylabel(fontsize=10)
    plt.show()
    """

    """
    try:
        heatmap = plt.imshow(all_time_df)
        plt.xticks(range(len(all_time_df.columns.values)), all_time_df.columns.values)
        # plt.yticks(range(len(all_time_df['Name'])), all_time_df['Name'])
        plt.yticks(range(len(all_time_df['Name'])), all_time_df['Name'])
        # cbar = plt.colorbar(mappable=heatmap, ticks=[0,1])
    except Exception as e:
        print('Problem with heatmap')
        print(e)
    """

    """

    all_time_df['set_1'] = all_time_df['Year'].between(1880, 1889, inclusive=True)
    set_1 = all_time_df.loc[all_time_df['set_1']==True]
    print('Set 1:')
    print(set_1)

    try:
        counts = set_1.to_numpy(dtype=int, copy=True)
        # counts = np.matrix(counts)
        # years = set_1['Year'].to_numpy(dtype=int, copy=True)
        # names = set_1['Name'].to_numpy(dtype=str, copy=True)
        
    except Exception as e:
        print('Failed to produce heatmap labels')
        print(e)

    try:
        def valid_imshow_data(data):
            data = np.asarray(data)
            if data.ndim == 2:
                return True
            elif data.ndim == 3:
                if 3 <= data.shape[2] <= 4:
                    return True
                else:
                    print('The "data" has 3 dimensions but the last dimension '
                        'must have a length of 3 (RGB) or 4 (RGBA), not "{}".'
                        ''.format(data.shape[2]))
                    return False
            else:
                print('To visualize an image the data must be 2 dimensional or '
                    '3 dimensional, not "{}".'
                    ''.format(data.ndim))
                return False
            
        valid_imshow_data(counts)
    except Exception as e:
        print('Check shape didn\'t work')
        print(e)

    try:
        fig, ax = plt.subplots()
        im = ax.imshow(counts)

        # Show all ticks and label them with the respective list entries
        # ax.set_xticks(np.arange(len(farmers)), labels=farmers)
        ax.set_xticks(years, labels=years)
        # ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)
        ax.set_yticks(names, labels=names)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
    """
    """
        try: 
            # Loop over data dimensions and create text annotations.
            for i in range(len(names)):
                for j in range(len(years)):
                    text = ax.text(j, i, counts[i, j],
                                ha="center", va="center", color="w")
        except Exception as e:
            print('Failed to produce heatmap labels')
            print(e)

        ax.set_title("Top-10 Baby Names per Year, US, 1880-2018")
        fig.tight_layout()
        plt.savefig('./outputs/Task_2.png')
        plt.show()
        
    except Exception as e:
        print('Failed to produce heatmap')
        print(e)
    """


def main():
    # load_name_data() # Running this line creates a feather binary of the full name dataframe
    load_feather()
    # task_1_a()
    # task_1_b()
    # task_1_c()
    task_2()

if __name__ == '__main__':
    main()

