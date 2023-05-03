import pandas as pd
from onlyEnergy import *
excel_data = pd.read_excel('/mnt/c/ПРоект/table.xlsx')
data = pd.DataFrame(excel_data, columns=['name'])
t=list(data['name'])
for fil in t:
    file=fil+'.pdb'
    E_in_file(file)

