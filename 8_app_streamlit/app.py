import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import gurobipy_pandas as gppd
from gurobi_ml import add_predictor_constr
import gurobipy as gp
from optimization_engine_v1 import optimization_engine
import plotly.graph_objects as go

################################# set page configuration #################################
st.set_page_config(layout="wide")



################################# Read env variables #################################
import os

""" Read env variables and save it as python variable """
WLSACCESSID = os.environ.get("WLSACCESSID", "")
WLSSECRET = os.environ.get("WLSSECRET", "")
LICENSEID = int(os.environ.get("LICENSEID", ""))
PROJECT_GCP = os.environ.get("PROJECT_GCP", "")

################################# LOAD LICENCE GUROBI - using env variables #################################
params = {
"WLSACCESSID": WLSACCESSID,
"WLSSECRET": WLSSECRET,
"LICENSEID": LICENSEID
}
env = gp.Env(params=params)


######################## ORDER CODES THAT SHOW INFORMATION IN THE UI ########################
if __name__ == "__main__":


    ######################## ------------------------------------- FORM TO INPUT VALUES OF OPTIMIZER - SIDEBAR ------------------------------------- ########################
    with st.form(key ='Form1'):
        with st.sidebar:
            st.header('----- INPUT PARAMS TO RUN OPTIMIZATION -----')
            
            ############## PARAMETERS OF OPTIMIZATION PROBLEM ##############
            st.divider()
            col1_sidebar, col2_sidebar, col3_sidebar = st.columns(3)

            ### COLUMN 1 - d0eop_microkappa
            col1_sidebar.write('**------ Parámetros D0EOP ------**')
            col1_sidebar.write('**Input VNC**')
            input_kappa_d0 = col1_sidebar.number_input("Kappa D0", min_value = 0.0, max_value = 10000.0, value = 6.34) # input_kappa_d0 = 6.349346  # "240AIT063A.PNT"
            input_brillo_d0 = col1_sidebar.number_input("Brillo D0", min_value = 0.0, max_value = 10000.0, value = 61.82) # input_brillo_d0 = 61.826925 # "240AIT063B.PNT"
            input_calc_prod_d0 = col1_sidebar.number_input("Prod D0", min_value = 0.0, max_value = 10000.0, value = 3346.85) #  = 3346.85825 # "calc_prod_d0"
            input_dqo_evaporadores = col1_sidebar.number_input("DQO", min_value = 0.0, max_value = 10000.0, value = 707.26) # input_dqo_evaporadores = 707.265937 # "SSTRIPPING015"
            input_concentracion_clo2_d0 = col1_sidebar.number_input("Concentración clo2 D0", min_value = 0.0, max_value = 10000.0, value = 11.51) # input_concentracion_clo2_d0 = 11.516543 #"S276PER002"
            input_ph_a = col1_sidebar.number_input("Ph A", min_value = 0.0, max_value = 10000.0, value = 2.93) # input_ph_a = 2.93846 # "240AIC022.MEAS"

            col1_sidebar.write('**Input VC**')
            input_actual_value_especifico_dioxido_d0 = col1_sidebar.number_input("especifico_dioxido_d0", min_value = 0.0, max_value = 10000.0, value = 6.46) # input_actual_value_especifico_dioxido_d0 = 6.46 #"240FY050.RO02"
            input_actual_value_especifico_oxigeno_eop = col1_sidebar.number_input("especifico_oxigeno_eop", min_value = 0.0, max_value = 10000.0, value = 1.08) # input_actual_value_especifico_oxigeno_eop = 1.08 #"240FY118B.RO01"
            input_actual_value_especifico_peroxido_eop = col1_sidebar.number_input("especifico_peroxido_eop", min_value = 0.0, max_value = 10000.0, value = 4.90) # input_actual_value_especifico_peroxido_eop = 4.90 #"240FY11PB.RO01"
            input_actual_value_especifico_soda_eop = col1_sidebar.number_input("especifico_soda_eop", min_value = 0.0, max_value = 10000.0, value = 8.84) # input_actual_value_especifico_soda_eop = 8.84 #"240FY107A.RO01"

            col1_sidebar.write('**Delta VC**')
            input_delta_especifico_dioxido_d0 = col1_sidebar.number_input("delta_especifico_dioxido_d0", min_value = 0.0, max_value = 5.0, value = 1.0) # input_delta_especifico_dioxido_d0 = 1
            input_delta_especifico_oxigeno_eop = col1_sidebar.number_input("delta_especifico_oxigeno_eop", min_value = 0.0, max_value = 5.0, value = 1.0) # input_delta_especifico_oxigeno_eop = 1
            input_delta_especifico_peroxido_eop = col1_sidebar.number_input("delta_especifico_peroxido_eop", min_value = 0.0, max_value = 5.0, value = 1.0) # input_delta_especifico_peroxido_eop = 1
            input_delta_especifico_soda_eop = col1_sidebar.number_input("delta_especifico_soda_eop", min_value = 0.0, max_value = 5.0, value = 1.0) # input_delta_especifico_soda_eop = 1


            ### COLUMN 2 - d1_brillo
            col2_sidebar.write('**------ Parámetros D1 ------**')
            col2_sidebar.write('**Input VNC**')
            input_blancura_d1 = col2_sidebar.number_input("blancura_d1", min_value = 0.0, max_value = 10000.0, value = 85.05) # input_blancura_d1 = 85.051634 #"240AIT225B.PNT"
            input_prod_bypass = col2_sidebar.number_input("prod_bypass", min_value = 0.0, max_value = 10000.0, value = 169.45) # input_prod_bypass = 169.45041 #"240FI108A.PNT"
            input_calc_prod_d1 = col2_sidebar.number_input("calc_prod_d1", min_value = 0.0, max_value = 10000.0, value = 3347.66) # input_calc_prod_d1 = 3347.664057 #"calc_prod_d1"
            input_temperatura_d1 = col2_sidebar.number_input("temperatura_d1", min_value = 0.0, max_value = 10000.0, value = 77.21) # input_temperatura_d1 = 77.212822 #"240TIT223.PNT"

            col2_sidebar.write('**Input VC**')
            input_actual_value_especifico_dioxido_d1 = col2_sidebar.number_input("especifico_dioxido_d1", min_value = 0.0, max_value = 10000.0, value = 2.89) # input_actual_value_especifico_dioxido_d1 = 2.89 #"240FY218.RO02"
            input_actual_value_especifico_acido_d1 = col2_sidebar.number_input("especifico_acido_d1", min_value = 0.0, max_value = 10000.0, value = 1.93) # input_actual_value_especifico_acido_d1 = 1.93 #"240FY210A.RO01"

            col2_sidebar.write('**Delta VC**')
            input_delta_especifico_dioxido_d1 = col2_sidebar.number_input("delta_especifico_dioxido_d1", min_value = 0.0, max_value = 5.0, value = 1.0) # input_delta_especifico_dioxido_d1 = 1
            input_delta_especifico_acido_d1 = col2_sidebar.number_input("delta_especifico_acido_d1", min_value = 0.0, max_value = 5.0, value = 1.0) # input_delta_especifico_acido_d1 = 1



            ### COLUMN 3 - d1_brillo
            col3_sidebar.write('**------ Parámetros P ------**')

            col3_sidebar.write('**Input VNC**')
            input_calc_prod_p = col3_sidebar.number_input("calc_prod_p", min_value = 0.0, max_value = 10000.0, value = 3346.18) # input_calc_prod_p = 3346.185561 #"calc_prod_p"
            input_ph_p = col3_sidebar.number_input("ph_p", min_value = 0.0, max_value = 10000.0, value = 10.38) # input_ph_p = 10.382279 #"240AIC324.MEAS"

            col3_sidebar.write('**Input VC**')
            input_actual_value_especifico_soda_p = col3_sidebar.number_input("especifico_soda_p", min_value = 0.0, max_value = 10000.0, value = 2.65) # input_actual_value_especifico_soda_p = 2.65 #"240FY312.RO01"
            input_actual_value_especifico_peroxido_p = col3_sidebar.number_input("especifico_peroxido_p ", min_value = 0.0, max_value = 10000.0, value = 0.78) # input_actual_value_especifico_peroxido_p = 0.78 #"240FY397.RO01"
            input_actual_value_especifico_acido_p = col3_sidebar.number_input("especifico_acido_p", min_value = 0.0, max_value = 10000.0, value = 2.01) # input_actual_value_especifico_acido_p = 2.01 #"240FY430.RO01"

            col3_sidebar.write('**Delta VC**')
            input_delta_especifico_soda_p = col3_sidebar.number_input("delta_especifico_soda_p", min_value = 0.0, max_value = 5.0, value = 1.0) # input_delta_especifico_soda_p = 1
            input_delta_especifico_peroxido_p = col3_sidebar.number_input("delta_especifico_peroxido_p", min_value = 0.0, max_value = 5.0, value = 1.0) # input_delta_especifico_peroxido_p = 1
            input_delta_especifico_acido_p = col3_sidebar.number_input("delta_especifico_acido_p", min_value = 0.0, max_value = 5.0, value = 1.0) # input_delta_especifico_acido_p = 1


            ############## SUBMIT BUTTON ##############
            submitted_opt = st.form_submit_button(label = 'Run Optimization')




    ######################## ------------------------- RUN OPTIMIZATION WHEN USER SEND THE NEW VALUES OF OPTIMIZATION ------------------------- ########################
    if submitted_opt:

        ############################################# generate input dataframe

        df_input_values = pd.DataFrame()

        ########## d0eop_microkappa
        # Actual values features no controlables #
        df_input_values["240AIT063A.PNT"] = [input_kappa_d0]
        df_input_values["240AIT063B.PNT"] = [input_brillo_d0]
        df_input_values["calc_prod_d0"] = [input_calc_prod_d0]
        df_input_values["SSTRIPPING015"] = [input_dqo_evaporadores]
        df_input_values["S276PER002"] = [input_concentracion_clo2_d0]
        df_input_values["240AIC022.MEAS"] = [input_ph_a]

        # Actual values features controlables
        df_input_values["240FY050.RO02"] = [input_actual_value_especifico_dioxido_d0]
        df_input_values["240FY118B.RO01"] = [input_actual_value_especifico_oxigeno_eop]
        df_input_values["240FY11PB.RO01"] = [input_actual_value_especifico_peroxido_eop]
        df_input_values["240FY107A.RO01"] = [input_actual_value_especifico_soda_eop]

        ########## d1_brillo
        # Actual values features no controlables
        df_input_values["240AIT225B.PNT"] = [input_blancura_d1]
        df_input_values["240FI108A.PNT"] = [input_prod_bypass]
        df_input_values["calc_prod_d1"] = [input_calc_prod_d1]
        df_input_values["240TIT223.PNT"] = [input_temperatura_d1]

        # Actual values features controlables
        df_input_values["240FY218.RO02"] = [input_actual_value_especifico_dioxido_d1]
        df_input_values["240FY210A.RO01"] = [input_actual_value_especifico_acido_d1]


        ########## p_blancura
        # Actual values features no controlables
        df_input_values["calc_prod_p"] = [input_calc_prod_p]
        df_input_values["240AIC324.MEAS"] = [input_ph_p]

        # Actual values features controlables
        df_input_values["240FY312.RO01"] = [input_actual_value_especifico_soda_p]
        df_input_values["240FY397.RO01"] = [input_actual_value_especifico_peroxido_p]
        df_input_values["240FY430.RO01"] = [input_actual_value_especifico_acido_p]




        ############################################# Load configuration file optimization engine
        # load bounds of decision variables in optimization that also are FEATURES in machine learning models
        path_bounds_decision_var_features = 'config/optimization_engine/config_optimization/Bounds-DecisionVar-Features-x.xlsx'
        bounds_decision_var_features = pd.read_excel(path_bounds_decision_var_features)

        # load bounds of decision variables in optimization that also are TARGETS in machine learning models
        # obs: the targets of one model are features in the next model. They are defined once
        path_bounds_decison_var_target = 'config/optimization_engine/config_optimization/Bounds-DecisionVar-Target-y.xlsx'
        bounds_decison_var_target = pd.read_excel(path_bounds_decison_var_target)


        ############## Load file with deltas for each decision variable ##### IMPORTANT - FOR THIS EXAMPLE THE USER INGRESS THE DELTAS VALUES
        path_deltas_decision_var_features = 'config/optimization_engine/config_optimization/Deltas-DecisionVar-Features-x.xlsx'
        deltas_decision_var_features = pd.read_excel(path_deltas_decision_var_features)

        ############## change values deltas acording input values deltas
        # d0eop
        deltas_decision_var_features.loc[deltas_decision_var_features['TAG_DESCRIPTION'] == 'especifico_dioxido_d0', 'DELTA'] = input_delta_especifico_dioxido_d0
        deltas_decision_var_features.loc[deltas_decision_var_features['TAG_DESCRIPTION'] == 'especifico_oxigeno_eop', 'DELTA'] = input_delta_especifico_oxigeno_eop
        deltas_decision_var_features.loc[deltas_decision_var_features['TAG_DESCRIPTION'] == 'especifico_peroxido_eop', 'DELTA'] = input_delta_especifico_peroxido_eop
        deltas_decision_var_features.loc[deltas_decision_var_features['TAG_DESCRIPTION'] == 'especifico_soda_eop', 'DELTA'] = input_delta_especifico_soda_eop

        # d1
        deltas_decision_var_features.loc[deltas_decision_var_features['TAG_DESCRIPTION'] == 'especifico_dioxido_d1', 'DELTA'] = input_delta_especifico_dioxido_d1
        deltas_decision_var_features.loc[deltas_decision_var_features['TAG_DESCRIPTION'] == 'especifico_acido_d1', 'DELTA'] = input_delta_especifico_acido_d1

        # p
        deltas_decision_var_features.loc[deltas_decision_var_features['TAG_DESCRIPTION'] == 'especifico_soda_p', 'DELTA'] = input_delta_especifico_soda_p
        deltas_decision_var_features.loc[deltas_decision_var_features['TAG_DESCRIPTION'] == 'especifico_peroxido_p', 'DELTA'] = input_delta_especifico_peroxido_p
        deltas_decision_var_features.loc[deltas_decision_var_features['TAG_DESCRIPTION'] == 'especifico_acido_p', 'DELTA'] = input_delta_especifico_acido_p

        # load prices
        path_prices = 'config/optimization_engine/config_optimization/price-chemicals.xlsx'
        prices = pd.read_excel(path_prices)



        ############################################# run optimization
        status_solver, opt_cost, solution, actual_values, diff_delta_decision_var = optimization_engine(df_input_values,
                                                                                                        bounds_decision_var_features,
                                                                                                        bounds_decison_var_target,
                                                                                                        deltas_decision_var_features,
                                                                                                        prices,
                                                                                                        env)






    ######################## ------------------------------------- INFO IN MAIN PAGE ------------------------------------- ########################
    

    # two tabs - first one show results - second show detail of model
    tab1, tab2 = st.tabs(["Results Optimization", "Details Optimization Modeling"])

    #### COLUMN1

    tab1.markdown("### ----- RESULTS OPTIMIZATION -----")
    if submitted_opt:
        if status_solver == 2: # optimal solution was founded
            
            # costs
            tab1.write(f"\n The optimal cost: ${opt_cost}")
            
            # solution
            tab1.write('SOLUTION')
            solution.columns = ['solution']
            actual_values.columns = ['actual_values']
            diff_delta_decision_var.columns = ['diff']
            final_solution = pd.concat([solution, actual_values, diff_delta_decision_var], axis = 1)
            tab1.dataframe(final_solution)

            # download solution
            final_solution = solution.to_csv()
            tab1.download_button("Download Optimal Solution", final_solution, file_name='final_solution.csv', key='csv_key')
            #os.remove('final_solution.csv')

            # plot change chemicals
            tab1.write('PLOT CHANGE CHEMICALS')
            categories = actual_values.index.tolist()
            fig_chemicals = go.Figure()
            fig_chemicals.add_trace(go.Scatterpolar(
                r = solution[:-3].values.squeeze(),
                theta = categories,
                fill='toself',
                name = 'SOLUTION'
            ))
            fig_chemicals.add_trace(go.Scatterpolar(
                r = actual_values.values.squeeze(),
                theta = categories,
                fill='toself',
                name = 'ACTUAL VALUES'
            ))
            st.plotly_chart(fig_chemicals)


        
        else:
            tab1.write("Model is infeasible or unbounded - Change the input parameters")
    
    
    else: # else if the user doesn't click the button submit in the form
        pass

    #### COLUMN2
    tab2.markdown("### ----- INFO OPTIMIZATION PROBLEM -----")
    tab2.write(" to do ")















