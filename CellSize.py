import pandas as pd
import matplotlib.pyplot as plt
import os

path = os.getcwd()
if os.path.exists(path):
    try:
        os.remove("Distances.txt")
    except:
        pass
file = pd.read_excel('data.xlsx')

all_y = [file[y] for y in file]
x = [x for x in all_y[1]]
for idx, value in enumerate(x):
    if value==6:
        idx1 = idx
    elif value == 10:
        idx2 = idx

  
def FindMin(y,x):
    '''
    A function to find the first and last global minima within distance value of < 6 and > 10  
    '''
    min_y1 = y[0]
    min_y2 = y[0]
    min_x1 = x[0]
    min_x2 = x[1]  
    for idx, x in enumerate(x):
        if idx < idx1 and y[idx] < min_y1:
            min_y1 = y[idx]
            min_x1 = x
        elif idx > idx2 and y[idx] < min_y2:
            min_y2 = y[idx]
            min_x2 = x

    final_distance = min_x2-min_x1
    with open('Distances.txt', 'a') as output:
        output.write(str(final_distance))
        output.write('\n')
        
    print('Distance is: ', final_distance,min_x1,min_x2, 'First min: ',min_y1,'Second min: ', min_y2)
    
for y in all_y[2:]:
    FindMin(y, x)
    
    # plt.figure(figsize=(10,10))   ####  uncomment these lines to visualize every plot
    # plt.style.use('seaborn')
    # plt.scatter(x,y,marker="o",s=30,edgecolors="black",c="yellow")
    # plt.title("Excel sheet to Scatter Plot")
    # plt.xlabel('Distance (µm)')
    # plt.ylabel('Intensity')
    # plt.show()

       
       
