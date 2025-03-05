#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pickle

#####functions#######
def maxwell_model(P_MOF,P_polymer,volume_fraction=0.15):
    P_MMM = P_polymer*((P_MOF+2*P_polymer-2*volume_fraction*(P_polymer-P_MOF))/(P_MOF+2*P_polymer+volume_fraction*(P_polymer-P_MOF)))
    return P_MMM

def transfer_predicted_y(y_predict):
    y_predict_actual=[]
    for i in range(len(y_predict)):
        if y_predict[i]>=0:
            y_predict_actual.append(y_predict[i])
        else:
            y_predict_actual.append(10**y_predict[i])
    return y_predict_actual

def get_P(temp_df, Loading_pressure):
    MOF_density = temp_df.iloc[0,-1] #kg/m3
    mol_D = temp_df.iloc[0,-4]#cm2/s
    #unit conversion of D
    mol_D = mol_D * 1e-4 #m2/s
    mol_L = temp_df.iloc[0,-2] #mol/kg
    mol_L = mol_L * MOF_density #mol/m3
    mol_Solubility = mol_L/Loading_pressure #mol/m3/Pa
    mol_P = mol_D*mol_Solubility #m2 mol/m3 Pa s
    return mol_P/3.348/1e-16 #barrer

# Taking user inputs
molecule = input("Enter molecule name (Available: H2, N2, O2, He, CO2, CH4): ")
mof = input("Enter MOF name (CSD codename): ")
polymer_permeability = float(input("Enter polymer permeability / Barrer (From exp or use Polymer Genome): "))
pressure = float(input("Enter pressure in Pascal: "))
volume_fraction = float(input("Enter volume fraction in MMM (default 0.15): "))

#####start calculation##############
#####first see if this MOF passes flexibility on D classification 
df_Ds_all_final=pd.read_csv('data/MOF_2019_notrelaxed_D_preds.csv')
if df_Ds_all_final[(df_Ds_all_final['molecule']==molecule) & (df_Ds_all_final['MOF'].str.startswith(mof))].empty:
    output = 'This MOF not in dataset or MOF flexibility has non-negligible influence on this MOF/molecule diffusion.'

else:
    D_prediction=df_Ds_all_final[(df_Ds_all_final['molecule']==molecule) & (df_Ds_all_final['MOF'].str.startswith(mof))]
    #######get L prediction of this MOF############

    #read KH predictions 
    DataMatrix_2019_K=pd.read_csv('data/MOF_2019_notrelaxed_K_preds.csv')
    DataMatrix_2019_K['logK']=np.log10(DataMatrix_2019_K['K_average'])

    #select the MOF and molecule line:
    if DataMatrix_2019_K[(DataMatrix_2019_K['molecule']==molecule) & (DataMatrix_2019_K['MOF'].str.startswith(mof))].empty:
        output = 'Henry\'s constant result not available.'
    else:
        DataMatrix_2019_K=DataMatrix_2019_K[(DataMatrix_2019_K['molecule']==molecule) & (DataMatrix_2019_K['MOF'].str.startswith(mof))]

        # Perform the merge using the 'key' column
        merged_K_unrelaxed_2019_data= DataMatrix_2019_K.copy()
        merged_K_unrelaxed_2019_data['pressure']=pressure

        #add langmuir descriptor
        merged_K_unrelaxed_2019_data['Langmuir_ratio_1']=merged_K_unrelaxed_2019_data['K_average']*merged_K_unrelaxed_2019_data['pressure']/(1+merged_K_unrelaxed_2019_data['K_average']*merged_K_unrelaxed_2019_data['pressure'])

        #choose the three molecules, and clean the columns
        descriptor_list=['PLD', 'LCD', 'VF', 'percentile-0.01', 'percentile-0.1',
               'percentile-0.25', 'percentile-0.5', 'percentile-0.75',
               'percentile-1.0', 'percentile-2.5', 'percentile-5.0', 'percentile-7.5',
               'percentile-10', 'percentile-25', 'mu_0.25-0.01', 'mu_0.25-0.1',
               'mu_0.25-0.25', 'mu_0.25-0.5', 'mu_0.25-0.75', 'mu_0.25-1.0',
               'mu_0.25-2.5', 'mu_0.25-5.0', 'mu_0.25-7.5', 'mu_0.25-10', 'mu_0.25-25',
               'mu_0.5-0.01', 'mu_0.5-0.1', 'mu_0.5-0.25', 'mu_0.5-0.5', 'mu_0.5-0.75',
               'mu_0.5-1.0', 'mu_0.5-2.5', 'mu_0.5-5.0', 'mu_0.5-7.5', 'mu_0.5-10',
               'mu_0.5-25', 'Tc', 'Pc', 'w', 'xlogP', 'diameter_1', 'diameter_2',
               'diameter_3', 'logK', 'Langmuir_ratio_1', 'pressure']

        L_unrelaxed_2019_data_cleaned=merged_K_unrelaxed_2019_data[['MOF', 'molecule', 'PLD', 'LCD',
               'VF', 'percentile-0.01', 'percentile-0.1', 'percentile-0.25',
               'percentile-0.5', 'percentile-0.75', 'percentile-1.0', 'percentile-2.5',
               'percentile-5.0', 'percentile-7.5', 'percentile-10', 'percentile-25',
               'mu_0.25-0.01', 'mu_0.25-0.1', 'mu_0.25-0.25', 'mu_0.25-0.5',
               'mu_0.25-0.75', 'mu_0.25-1.0', 'mu_0.25-2.5', 'mu_0.25-5.0',
               'mu_0.25-7.5', 'mu_0.25-10', 'mu_0.25-25', 'mu_0.5-0.01', 'mu_0.5-0.1',
               'mu_0.5-0.25', 'mu_0.5-0.5', 'mu_0.5-0.75', 'mu_0.5-1.0', 'mu_0.5-2.5',
               'mu_0.5-5.0', 'mu_0.5-7.5', 'mu_0.5-10', 'mu_0.5-25', 'Tc', 'Pc', 'w',
               'xlogP', 'diameter_1', 'diameter_2', 'diameter_3', 'logK','Langmuir_ratio_1', 'pressure']].copy()


        for i in range(10):
            temp_regressor = pickle.load(open('ML_models/c_ML/c_reg_{}.pkl'.format(i),'rb'))
            L_unrelaxed_2019_data_cleaned['predict_L_{}'.format(i)]=transfer_predicted_y(temp_regressor.predict(L_unrelaxed_2019_data_cleaned[descriptor_list].values))


        L_unrelaxed_2019_data_cleaned['L_average']=np.mean(L_unrelaxed_2019_data_cleaned.iloc[:,48:58],axis=1)
        L_unrelaxed_2019_data_cleaned['L_std']=np.std(L_unrelaxed_2019_data_cleaned.iloc[:,48:58],axis=1)

        #############start P calculations#########
        L_D=pd.merge(D_prediction,L_unrelaxed_2019_data_cleaned[['MOF','molecule','pressure','L_average']],on=['MOF','molecule'],how='left')

        #get density of the MOFs
        mof_density=pd.read_csv('data/2019_not_relaxed_density.txt',sep='\t',header=None)
        mof_density.columns=['MOF','Density']
        mof_density[['MOF','other']] = mof_density.iloc[:,0].str.split('_volpo',expand=True)
        mof_density=mof_density.drop(columns=['other'])
        mof_density[['MOF','other']] = mof_density.iloc[:,0].str.split('.txt',expand=True)
        mof_density=mof_density.drop(columns=['other'])
        mof_density['Density/kg/m3']=mof_density['Density']*1000

        L_D_merged=pd.merge(L_D,mof_density[['MOF','Density/kg/m3']],how='left',on='MOF')
        if not L_D_merged[L_D_merged.duplicated()].empty:
            L_D_cleaned=L_D_merged.drop(index=L_D_merged[L_D_merged.duplicated()].index).reset_index(drop=True).copy()
        else:
            L_D_cleaned=L_D_merged.copy()

        mof_p=get_P(L_D_cleaned, pressure)
        print('MOF permeability / Barrer', mof_p)

        output = maxwell_model(mof_p,polymer_permeability,volume_fraction)

# Display the result
print("MMM permeability of this molecule / Barrer:", output)

if __name__ == "__main__":
    pass
