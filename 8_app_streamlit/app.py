import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
import os

#gurobi
import gurobipy_pandas as gppd
from gurobi_ml import add_predictor_constr
import gurobipy as gp


################################# set page configuration #################################
st.set_page_config(layout="wide")



######################## FUNCTIONS USES TO RUN THE APP ########################


@st.cache_resource(show_spinner="Loading models...")
def model_d0eop_microkappa():
    path_model_d0eop_microkappa = f'models/d0eop_microkappa/lr.pkl'
    model_d0eop_microkappa = pd.read_pickle(path_model_d0eop_microkappa)
    return model_d0eop_microkappa


@st.cache_resource(show_spinner="Loading models...")
def model_d1_brillo():
    path_model_d1_brillo = f'models/d1_brillo/lr.pkl'
    model_d1_brillo = pd.read_pickle(path_model_d1_brillo)
    return model_d1_brillo



@st.cache_resource(show_spinner="Loading models...")
def model_p_blancura():
    path_model_p_blancura = f'models/p_blancura/lr.pkl'
    model_p_blancura = pd.read_pickle(path_model_p_blancura)
    return model_p_blancura



######################## ORDER CODES THAT SHOW INFORMATION IN THE UI ########################
if __name__ == "__main__":


    ######################## FORM TO INPUT VALUES OF OPTIMIZER - SIDEBAR ########################
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




    ######################## RUN OPTIMIZATION WHEN USER SEND THE NEW VALUES OF OPTIMIZATION ########################
    if submitted_opt:

        ##################### 2. Load model machine learning - MANUALLY - HARDCODED #####################
        model_d0eop_microkappa = model_d0eop_microkappa()
        model_d1_brillo = model_d1_brillo()
        model_p_blancura = model_p_blancura()


        ##################### 3. Define list of features and target for each model
        ######################## model d0eop_microkappa ########################

        list_features_d0eop_microkappa = [
            "240AIT063A.PNT", #kappa_d0
            "240AIT063B.PNT", #brillo_d0
            "calc_prod_d0", #calc_prod_d0
            "240FY050.RO02", #especifico_dioxido_d0 - VC
            "SSTRIPPING015", #dqo_evaporadores
            "S276PER002", #concentracion_clo2_d0
            "240AIC022.MEAS", #ph_a
            "240FY118B.RO01", #especifico_oxigeno_eop - VC
            "240FY11PB.RO01", #especifico_peroxido_eop - VC
            "240FY107A.RO01", #especifico_soda_eop - VC
        ]

        list_features_controlables_d0eop_microkappa = [
            "240FY050.RO02", #especifico_dioxido_d0 - VC,
            "240FY118B.RO01", #especifico_oxigeno_eop - VC
            "240FY11PB.RO01", #especifico_peroxido_eop - VC
            "240FY107A.RO01", #especifico_soda_eop - VC
        ]

        list_target_d0eop_microkappa = ['240AIT225A.PNT'] # microkappa_d1

        ######################## model d1_brillo ########################

        list_features_d1_brillo = [
            "240AIT225B.PNT", # blancura_d1
            "240AIT225A.PNT", # microkappa_d1
            "240FI108A.PNT", # prod_bypass
            "calc_prod_d1", #calc_prod_d1
            "240FY218.RO02", #especifico_dioxido_d1 - VC
            "240TIT223.PNT", #temperatura_d1
            "240FY210A.RO01", # especifico_acido_d1 - vc
        ]

        list_features_controlables_d1_brillo = [
            "240FY218.RO02", #especifico_dioxido_d1 - VC
            "240FY210A.RO01", # especifico_acido_d1 - vc
        ]

        list_target_d1_brillo = ['240AIT322B.PNT'] # brillo_entrada_p

        ######################## model p_blancura ########################

        list_features_p_blancura = [
            "240AIT322B.PNT", #brillo_p
            "calc_prod_p", #calc_prod_p
            "240FY312.RO01", # especifico_soda_p - VC
            "240FY397.RO01", #especifico_peroxido_p - VC
            "240AIC324.MEAS", #ph_p,
            "240FY430.RO01", #especifico_acido_p - VC
        ]

        list_features_controlables_p_blancura = [    
            "240FY312.RO01", # especifico_soda_p - VC
            "240FY397.RO01", #especifico_peroxido_p - VC
            "240FY430.RO01", #especifico_acido_p - VC
        ]

        list_target_p_blancura = ['240AIT416B.PNT'] #blancura_salida_p



        ##################### 4. Load tables parameters for optimization - bounds
        path_bounds_decision_var_features = 'config/Optimization-Bounds-DecisionVar.xlsx'
        bounds_decision_var_features = pd.read_excel(path_bounds_decision_var_features)

        path_bounds_decison_var_target = 'config/Optimization-Bounds-Target.xlsx'
        bounds_decison_var_target = pd.read_excel(path_bounds_decison_var_target)





        ##################### 5. Load Prices
        path_prices = 'config/price-chemicals.xlsx'
        prices = pd.read_excel(path_prices)



        ########################################## RUN OPTIMIZATION ##########################################
        ##################### 0. Load transversal params - sets of optimization model

        list_bleaching = ['bleaching']
        index_bleaching = pd.Index(list_bleaching)



        ##################### 1. Create guroby optimization model
        m = gp.Model(name = "Bleaching Optimization v2")


        ##################### 3. Input parameters of optimization model
        #3.1 Actual values of decision variables
        ######################## actual values for model d0eop_microkappa ########################
        ### IMPORTANTE ASEGURSE QUE EL TAG CORRESPONDE CON EL NOMBRE DEFINIDO


        #especifico_dioxido_d0
        actual_value_especifico_dioxido_d0 = input_actual_value_especifico_dioxido_d0

        #especifico_oxigeno_eop
        actual_value_especifico_oxigeno_eop = input_actual_value_especifico_oxigeno_eop

        #especifico_peroxido_eop
        actual_value_especifico_peroxido_eop = input_actual_value_especifico_peroxido_eop

        #especifico_soda_eop
        actual_value_especifico_soda_eop = input_actual_value_especifico_soda_eop

        ######################## actual values for model d1_brillo ########################

        #especifico_dioxido_d1
        actual_value_especifico_dioxido_d1 = input_actual_value_especifico_dioxido_d1

        #especifico_acido_d1
        actual_value_especifico_acido_d1 = input_actual_value_especifico_acido_d1

        ######################## actual values for model p_blancura ########################


        #especifico_soda_p
        actual_value_especifico_soda_p = input_actual_value_especifico_soda_p


        #especifico_peroxido_p
        actual_value_especifico_peroxido_p = input_actual_value_especifico_peroxido_p


        #especifico_acido_p
        actual_value_especifico_acido_p = input_actual_value_especifico_acido_p


        # 3.2 Parameters Rate of Change decision variables
        ######################## actual values for model d0eop_microkappa ########################

        #especifico_dioxido_d0
        delta_especifico_dioxido_d0 = input_delta_especifico_dioxido_d0

        #especifico_oxigeno_eop
        delta_especifico_oxigeno_eop = input_delta_especifico_oxigeno_eop

        #especifico_peroxido_eop
        delta_especifico_peroxido_eop = input_delta_especifico_peroxido_eop

        #especifico_soda_eop
        delta_especifico_soda_eop = input_delta_especifico_soda_eop

        ######################## actual values for model d1_brillo ########################

        #especifico_dioxido_d1
        delta_especifico_dioxido_d1 = input_delta_especifico_dioxido_d1

        #especifico_acido_d1
        delta_especifico_acido_d1 = input_delta_especifico_acido_d1

        ######################## actual values for model p_blancura ########################

        #especifico_soda_p
        delta_especifico_soda_p = input_delta_especifico_soda_p


        #especifico_peroxido_p
        delta_especifico_peroxido_p = input_delta_especifico_peroxido_p

        #especifico_acido_p
        delta_especifico_acido_p = input_delta_especifico_acido_p




        ##################### 4. Features input machine learning model fixed (that are not decision variables or parameters in optimization model)
        ######################## generate instance for model d0eop_microkappa ########################

        # list feature NC
        list_features_d0eop_microkappa_no_vc = list(set(list_features_d0eop_microkappa) - set(list_features_controlables_d0eop_microkappa))

        # generate dataframe with the mean
        #instance_no_controlables_d0eop_microkappa = data[list_features_d0eop_microkappa_no_vc].mean().to_frame().T
        instance_no_controlables_d0eop_microkappa = pd.DataFrame()
        instance_no_controlables_d0eop_microkappa["240AIT063A.PNT"] = [input_kappa_d0]
        instance_no_controlables_d0eop_microkappa["240AIT063B.PNT"] = [input_brillo_d0]
        instance_no_controlables_d0eop_microkappa["calc_prod_d0"] = [input_calc_prod_d0]
        instance_no_controlables_d0eop_microkappa["SSTRIPPING015"] = [input_dqo_evaporadores]
        instance_no_controlables_d0eop_microkappa["S276PER002"] = [input_concentracion_clo2_d0]
        instance_no_controlables_d0eop_microkappa["240AIC022.MEAS"] = [input_ph_a]


        ######################## generate instance for model d1_brillo ########################

        # list features NC
        list_features_d1_brillo_no_vc = list(set(list_features_d1_brillo) - set(list_features_controlables_d1_brillo)) # substract vc d1_brillo
        list_features_d1_brillo_no_vc = list(set(list_features_d1_brillo_no_vc) - set(list_target_d0eop_microkappa))# substract target d0eop_microkappa

        # generate dataframe with the mean
        #instance_no_controlables_d1_brillo = data[list_features_d1_brillo_no_vc].mean().to_frame().T
        instance_no_controlables_d1_brillo = pd.DataFrame()
        instance_no_controlables_d1_brillo["240AIT225B.PNT"] = [input_blancura_d1]
        instance_no_controlables_d1_brillo["240FI108A.PNT"] = [input_prod_bypass]
        instance_no_controlables_d1_brillo["calc_prod_d1"] = [input_calc_prod_d1]
        instance_no_controlables_d1_brillo["240TIT223.PNT"] = [input_temperatura_d1]


        ######################## generate instance for model p_blancura ########################

        # list features NC
        list_features_p_blancura_no_vc = list(set(list_features_p_blancura) - set(list_features_controlables_p_blancura)) # substract vc p_blancura
        list_features_p_blancura_no_vc = list(set(list_features_p_blancura_no_vc) - set(list_target_d1_brillo))# substract target d1_brillo

        # generate dataframe with the mean
        #instance_no_controlables_p_blancura = data[list_features_p_blancura_no_vc].mean().to_frame().T
        instance_no_controlables_p_blancura = pd.DataFrame()
        instance_no_controlables_p_blancura["calc_prod_p"] = [input_calc_prod_p]
        instance_no_controlables_p_blancura["240AIC324.MEAS"] = [input_ph_p]



        ##################### 5. Decision variables of optimization model
        ######################## decision variables that are FEATURES in Machiine Learning Models ########################

        # model d0eop_microkappa
        especifico_dioxido_d0 = gppd.add_vars(m, index_bleaching, name = "especifico_dioxido_d0", 
                                            lb = bounds_decision_var_features[bounds_decision_var_features['TAG_DESCRIPTION'] == 'especifico_dioxido_d0']['MIN_VALUE'].values[0], 
                                            ub = bounds_decision_var_features[bounds_decision_var_features['TAG_DESCRIPTION'] == 'especifico_dioxido_d0']['MAX_VALUE'].values[0]
                                            )

        especifico_oxigeno_eop = gppd.add_vars(m, index_bleaching, name = "especifico_oxigeno_eop", 
                                            lb = bounds_decision_var_features[bounds_decision_var_features['TAG_DESCRIPTION'] == 'especifico_oxigeno_eop']['MIN_VALUE'].values[0], 
                                            ub = bounds_decision_var_features[bounds_decision_var_features['TAG_DESCRIPTION'] == 'especifico_oxigeno_eop']['MAX_VALUE'].values[0]
                                            )

        especifico_peroxido_eop = gppd.add_vars(m, index_bleaching, name = "especifico_peroxido_eop", 
                                            lb = bounds_decision_var_features[bounds_decision_var_features['TAG_DESCRIPTION'] == 'especifico_peroxido_eop']['MIN_VALUE'].values[0], 
                                            ub = bounds_decision_var_features[bounds_decision_var_features['TAG_DESCRIPTION'] == 'especifico_peroxido_eop']['MAX_VALUE'].values[0]
                                            )

        especifico_soda_eop = gppd.add_vars(m, index_bleaching, name = "especifico_soda_eop", 
                                            lb = bounds_decision_var_features[bounds_decision_var_features['TAG_DESCRIPTION'] == 'especifico_soda_eop']['MIN_VALUE'].values[0], 
                                            ub = bounds_decision_var_features[bounds_decision_var_features['TAG_DESCRIPTION'] == 'especifico_soda_eop']['MAX_VALUE'].values[0]
                                            )



        # model d1_brillo
        especifico_dioxido_d1 = gppd.add_vars(m, index_bleaching, name = "especifico_dioxido_d1", 
                                            lb = bounds_decision_var_features[bounds_decision_var_features['TAG_DESCRIPTION'] == 'especifico_dioxido_d1']['MIN_VALUE'].values[0], 
                                            ub = bounds_decision_var_features[bounds_decision_var_features['TAG_DESCRIPTION'] == 'especifico_dioxido_d1']['MAX_VALUE'].values[0]
                                            )

        especifico_acido_d1 = gppd.add_vars(m, index_bleaching, name = "especifico_acido_d1", 
                                            lb = bounds_decision_var_features[bounds_decision_var_features['TAG_DESCRIPTION'] == 'especifico_acido_d1']['MIN_VALUE'].values[0], 
                                            ub = bounds_decision_var_features[bounds_decision_var_features['TAG_DESCRIPTION'] == 'especifico_acido_d1']['MAX_VALUE'].values[0]
                                            )



        # model p_blancura
        especifico_soda_p = gppd.add_vars(m, index_bleaching, name = "especifico_soda_p", 
                                            lb = bounds_decision_var_features[bounds_decision_var_features['TAG_DESCRIPTION'] == 'especifico_soda_p']['MIN_VALUE'].values[0], 
                                            ub = bounds_decision_var_features[bounds_decision_var_features['TAG_DESCRIPTION'] == 'especifico_soda_p']['MAX_VALUE'].values[0]
                                            )

        especifico_peroxido_p = gppd.add_vars(m, index_bleaching, name = "especifico_peroxido_p", 
                                            lb = bounds_decision_var_features[bounds_decision_var_features['TAG_DESCRIPTION'] == 'especifico_peroxido_p']['MIN_VALUE'].values[0], 
                                            ub = bounds_decision_var_features[bounds_decision_var_features['TAG_DESCRIPTION'] == 'especifico_peroxido_p']['MAX_VALUE'].values[0]
                                            )

        especifico_acido_p = gppd.add_vars(m, index_bleaching, name = "especifico_acido_p", 
                                            lb = bounds_decision_var_features[bounds_decision_var_features['TAG_DESCRIPTION'] == 'especifico_acido_p']['MIN_VALUE'].values[0], 
                                            ub = bounds_decision_var_features[bounds_decision_var_features['TAG_DESCRIPTION'] == 'especifico_acido_p']['MAX_VALUE'].values[0]
                                            )


        ######################## decision variables that are TARGETS in Machiine Learning Models ########################

        # model d0eop_microkappa
        microkappa_d1 = gppd.add_vars(m, index_bleaching, name = "microkappa_d1", 
                                            lb = bounds_decison_var_target[bounds_decison_var_target['TAG_DESCRIPTION'] == 'microkappa_d1']['MIN_VALUE'].values[0], 
                                            ub = bounds_decison_var_target[bounds_decison_var_target['TAG_DESCRIPTION'] == 'microkappa_d1']['MAX_VALUE'].values[0]
                                            )

        # model d1_brillo
        brillo_entrada_p = gppd.add_vars(m, index_bleaching, name = "brillo_entrada_p", 
                                            lb = bounds_decison_var_target[bounds_decison_var_target['TAG_DESCRIPTION'] == 'brillo_entrada_p']['MIN_VALUE'].values[0], 
                                            ub = bounds_decison_var_target[bounds_decison_var_target['TAG_DESCRIPTION'] == 'brillo_entrada_p']['MAX_VALUE'].values[0]
                                            )
        # model p_blancura
        blancura_salida_p = gppd.add_vars(m, index_bleaching, name = "blancura_salida_p", 
                                            lb = bounds_decison_var_target[bounds_decison_var_target['TAG_DESCRIPTION'] == 'blancura_salida_p']['MIN_VALUE'].values[0], 
                                            ub = bounds_decison_var_target[bounds_decison_var_target['TAG_DESCRIPTION'] == 'blancura_salida_p']['MAX_VALUE'].values[0]
                                            )


        ############ decision variables that represent the difference between actual value FEATURES in Machine Learning Models and optimal value ############


        # model d0eop_microkappa
        diff_especifico_dioxido_d0 = gppd.add_vars(m, index_bleaching, name = "diff_especifico_dioxido_d0", 
                                            lb = -gp.GRB.INFINITY,
                                            ub = gp.GRB.INFINITY
                                            )

        diff_especifico_oxigeno_eop = gppd.add_vars(m, index_bleaching, name = "diff_especifico_oxigeno_eop", 
                                            lb = -gp.GRB.INFINITY,
                                            ub = gp.GRB.INFINITY
                                            )

        diff_especifico_peroxido_eop = gppd.add_vars(m, index_bleaching, name = "diff_especifico_peroxido_eop", 
                                            lb = -gp.GRB.INFINITY,
                                            ub = gp.GRB.INFINITY
                                            )

        diff_especifico_soda_eop = gppd.add_vars(m, index_bleaching, name = "diff_especifico_soda_eop", 
                                            lb = -gp.GRB.INFINITY,
                                            ub = gp.GRB.INFINITY
                                            )



        # model d1_brillo
        diff_especifico_dioxido_d1 = gppd.add_vars(m, index_bleaching, name = "diff_especifico_dioxido_d1", 
                                            lb = -gp.GRB.INFINITY,
                                            ub = gp.GRB.INFINITY
                                            )

        diff_especifico_acido_d1 = gppd.add_vars(m, index_bleaching, name = "diff_especifico_acido_d1", 
                                            lb = -gp.GRB.INFINITY,
                                            ub = gp.GRB.INFINITY
                                            )



        # model p_blancura
        diff_especifico_soda_p = gppd.add_vars(m, index_bleaching, name = "diff_especifico_soda_p", 
                                            lb = -gp.GRB.INFINITY,
                                            ub = gp.GRB.INFINITY
                                            )

        diff_especifico_peroxido_p = gppd.add_vars(m, index_bleaching, name = "diff_especifico_peroxido_p", 
                                            lb = -gp.GRB.INFINITY,
                                            ub = gp.GRB.INFINITY
                                            )

        diff_especifico_acido_p = gppd.add_vars(m, index_bleaching, name = "diff_especifico_acido_p", 
                                            lb = -gp.GRB.INFINITY,
                                            ub = gp.GRB.INFINITY
                                            )




        ##################### 6. Constraints (constraints that are not generated by a ml model)
        ######################## decision variables that are FEATURES in Machiine Learning Models ########################

        ###### model d0eop_microkappa ######
        # especifico_dioxido_d0
        m.addConstr(diff_especifico_dioxido_d0[0] >= (especifico_dioxido_d0[0] - actual_value_especifico_dioxido_d0), name = 'diff_especifico_dioxido_d0 positive segment')
        m.addConstr(diff_especifico_dioxido_d0[0] >= -(especifico_dioxido_d0[0] - actual_value_especifico_dioxido_d0), name = 'diff_especifico_dioxido_d0 negative segment')
        m.addConstr(diff_especifico_dioxido_d0[0] <= delta_especifico_dioxido_d0, name = 'diff_especifico_dioxido_d0 delta')


        # especifico_oxigeno_eop
        m.addConstr(diff_especifico_oxigeno_eop[0] >= (especifico_oxigeno_eop[0] - actual_value_especifico_oxigeno_eop), name = 'diff_especifico_oxigeno_eop positive segment')
        m.addConstr(diff_especifico_oxigeno_eop[0] >= -(especifico_oxigeno_eop[0] - actual_value_especifico_oxigeno_eop), name = 'diff_especifico_oxigeno_eop negative segment')
        m.addConstr(diff_especifico_oxigeno_eop[0] <= delta_especifico_oxigeno_eop, name = 'diff_especifico_oxigeno_eop delta')


        # especifico_peroxido_eop
        m.addConstr(diff_especifico_peroxido_eop[0] >= (especifico_peroxido_eop[0] - actual_value_especifico_peroxido_eop), name = 'diff_especifico_peroxido_eop positive segment')
        m.addConstr(diff_especifico_peroxido_eop[0] >= -(especifico_peroxido_eop[0] - actual_value_especifico_peroxido_eop), name = 'diff_especifico_peroxido_eop negative segment')
        m.addConstr(diff_especifico_peroxido_eop[0] <= delta_especifico_peroxido_eop, name = 'diff_especifico_peroxido_eop delta')


        # especifico_soda_eop
        m.addConstr(diff_especifico_soda_eop[0] >= (especifico_soda_eop[0] - actual_value_especifico_soda_eop), name = 'diff_especifico_soda_eop positive segment')
        m.addConstr(diff_especifico_soda_eop[0] >= -(especifico_soda_eop[0] - actual_value_especifico_soda_eop), name = 'diff_especifico_soda_eop negative segment')
        m.addConstr(diff_especifico_soda_eop[0] <= delta_especifico_soda_eop, name = 'diff_especifico_soda_eop delta')


        ###### model d1_brillo ######
        # especifico_dioxido_d1
        m.addConstr(diff_especifico_dioxido_d1[0] >= (especifico_dioxido_d1[0] - actual_value_especifico_dioxido_d1), name = 'diff_especifico_dioxido_d1 positive segment')
        m.addConstr(diff_especifico_dioxido_d1[0] >= -(especifico_dioxido_d1[0] - actual_value_especifico_dioxido_d1), name = 'diff_especifico_dioxido_d1 negative segment')
        m.addConstr(diff_especifico_dioxido_d1[0] <= delta_especifico_dioxido_d1, name = 'diff_especifico_dioxido_d1 delta')


        # especifico_acido_d1
        m.addConstr(diff_especifico_acido_d1[0] >= (especifico_acido_d1[0] - actual_value_especifico_acido_d1), name = 'diff_especifico_acido_d1 positive segment')
        m.addConstr(diff_especifico_acido_d1[0] >= -(especifico_acido_d1[0] - actual_value_especifico_acido_d1), name = 'diff_especifico_acido_d1 negative segment')
        m.addConstr(diff_especifico_acido_d1[0] <= delta_especifico_acido_d1, name = 'diff_especifico_acido_d1 delta')


        ###### model p_blancura ######
        # especifico_soda_p
        m.addConstr(diff_especifico_soda_p[0] >= (especifico_soda_p[0] - actual_value_especifico_soda_p), name = 'diff_especifico_soda_p positive segment')
        m.addConstr(diff_especifico_soda_p[0] >= -(especifico_soda_p[0] - actual_value_especifico_soda_p), name = 'diff_especifico_soda_p negative segment')
        m.addConstr(diff_especifico_soda_p[0] <= delta_especifico_soda_p, name = 'diff_especifico_soda_p delta')


        # especifico_peroxido_p
        m.addConstr(diff_especifico_peroxido_p[0] >= (especifico_peroxido_p[0] - actual_value_especifico_peroxido_p), name = 'diff_especifico_peroxido_p positive segment')
        m.addConstr(diff_especifico_peroxido_p[0] >= -(especifico_peroxido_p[0] - actual_value_especifico_peroxido_p), name = 'diff_especifico_peroxido_p negative segment')
        m.addConstr(diff_especifico_peroxido_p[0] <= delta_especifico_peroxido_p, name = 'diff_especifico_peroxido_p delta')


        # especifico_acido_p
        m.addConstr(diff_especifico_acido_p[0] >= (especifico_acido_p[0] - actual_value_especifico_acido_p), name = 'diff_especifico_acido_p positive segment')
        m.addConstr(diff_especifico_acido_p[0] >= -(especifico_acido_p[0] - actual_value_especifico_acido_p), name = 'diff_especifico_acido_p negative segment')
        m.addConstr(diff_especifico_acido_p[0] <= delta_especifico_acido_p, name = 'diff_especifico_acido_p delta')

        # update model
        m.update()




        ##################### 7. Add constraints that are machine learning models
        # 7.1 d0eop_microkappa
        # create instance with controlables variables. sorted according the list of features. ES MUY IMPORTANTE QUE ESTÉ ORDENADO LAS VARIABLES DE DECUISIÓN DE ACUERDO A LA LISTA DE FEATURES
        instance_controlables_d0eop_microkappa = pd.DataFrame([especifico_dioxido_d0, especifico_oxigeno_eop, especifico_peroxido_eop, especifico_soda_eop]).T
        instance_controlables_d0eop_microkappa.columns = list_features_controlables_d0eop_microkappa # rename columns
        instance_controlables_d0eop_microkappa.reset_index(inplace = True)
        instance_controlables_d0eop_microkappa.drop(columns = 'index', inplace = True)

        # append features controlables with no controlables
        instance_d0eop_microkappa = pd.concat([instance_no_controlables_d0eop_microkappa, instance_controlables_d0eop_microkappa], axis = 1)
        instance_d0eop_microkappa = instance_d0eop_microkappa[list_features_d0eop_microkappa] # sort features

        # set index - optimization set
        instance_d0eop_microkappa.index = index_bleaching

        ###### load ml constraint ######
        pred_constr_d0eop_microkappa = add_predictor_constr(gp_model = m, 
                                        predictor = model_d0eop_microkappa, 
                                        input_vars = instance_d0eop_microkappa, 
                                        output_vars = microkappa_d1,
                                        name = f'model_predict_d0eop_microkappa'
                                        )



        # 7.2 d1_brillo
        # (reemplazar "d0eop_microkappa" por "d1_brillo")
        ######################## instance model d1_brillo ########################

        # create instance with controlables variables. sorted according the list of features. ES MUY IMPORTANTE QUE ESTÉ ORDENADO LAS VARIABLES DE DECUISIÓN DE ACUERDO A LA LISTA DE FEATURES
        instance_controlables_d1_brillo = pd.DataFrame([especifico_dioxido_d1, especifico_acido_d1]).T # <---- change ---------<--------<--------
        instance_controlables_d1_brillo.columns = list_features_controlables_d1_brillo # rename columns
        instance_controlables_d1_brillo.reset_index(inplace = True)
        instance_controlables_d1_brillo.drop(columns = 'index', inplace = True)

        # create instance with target of previos model
        instance_previos_target_d1_brillo = pd.DataFrame([microkappa_d1]).T # <---- change ---------<--------<--------
        instance_previos_target_d1_brillo.columns  = list_target_d0eop_microkappa # rename columns
        instance_previos_target_d1_brillo.reset_index(inplace = True)
        instance_previos_target_d1_brillo.drop(columns = 'index', inplace = True)

        # append features controlables with no controlables
        instance_d1_brillo = pd.concat([instance_no_controlables_d1_brillo, instance_controlables_d1_brillo, instance_previos_target_d1_brillo], axis = 1)
        instance_d1_brillo = instance_d1_brillo[list_features_d1_brillo] # sort features

        # set index - optimization set
        instance_d1_brillo.index = index_bleaching

        ###### load ml constraint ######
        pred_constr_d1_brillo = add_predictor_constr(gp_model = m, 
                                        predictor = model_d1_brillo, 
                                        input_vars = instance_d1_brillo, 
                                        output_vars = brillo_entrada_p,
                                        name = f'model_predict_d1_brillo'
                                        )


        # 7.3 model p_blancura
        # (reemplazar "d1_brillo" por "p_blancura")
        ######################## instance model p_blancura ########################

        # create instance with controlables variables. sorted according the list of features. ES MUY IMPORTANTE QUE ESTÉ ORDENADO LAS VARIABLES DE DECUISIÓN DE ACUERDO A LA LISTA DE FEATURES
        instance_controlables_p_blancura = pd.DataFrame([especifico_soda_p, especifico_peroxido_p, especifico_acido_p]).T # <---- change ---------<--------<--------
        instance_controlables_p_blancura.columns = list_features_controlables_p_blancura # rename columns
        instance_controlables_p_blancura.reset_index(inplace = True)
        instance_controlables_p_blancura.drop(columns = 'index', inplace = True)

        # create instance with target of previos model
        instance_previos_target_p_blancura = pd.DataFrame([brillo_entrada_p]).T # <---- change ---------<--------<--------
        instance_previos_target_p_blancura.columns  = list_target_d1_brillo # rename columns
        instance_previos_target_p_blancura.reset_index(inplace = True)
        instance_previos_target_p_blancura.drop(columns = 'index', inplace = True)

        # append features controlables with no controlables
        instance_p_blancura = pd.concat([instance_no_controlables_p_blancura, instance_controlables_p_blancura, instance_previos_target_p_blancura], axis = 1)
        instance_p_blancura = instance_p_blancura[list_features_p_blancura] # sort features

        # set index - optimization set
        instance_p_blancura.index = index_bleaching


        ###### load ml constraint ######
        pred_constr_p_blancura = add_predictor_constr(gp_model = m, 
                                        predictor = model_p_blancura, 
                                        input_vars = instance_p_blancura, 
                                        output_vars = blancura_salida_p,
                                        name = f'model_predict_p_blancura'
                                        )




        ### 8. Define Objetive Function
        # ######################## define variable of costs of each stage ########################
        costs_d0 = especifico_dioxido_d0*prices['dioxido'].values[0]
        costs_eop = especifico_soda_eop*prices['soda'].values[0] + especifico_peroxido_eop*prices['peroxido'].values[0] + especifico_oxigeno_eop*prices['oxigeno'].values[0]
        costs_d1 = especifico_acido_d1*prices['acido'].values[0] + especifico_dioxido_d1*prices['dioxido'].values[0]
        costs_p = especifico_acido_p*prices['acido'].values[0] + especifico_soda_p*prices['soda'].values[0] + especifico_peroxido_p*prices['peroxido'].values[0]


        ######################## set objetive minimize costs ########################

        # it is necesary define with .sum() to get a guroli linear expression
        m.setObjective(costs_d0.sum()+ costs_eop.sum() + costs_d1.sum() + costs_p.sum(),
                    gp.GRB.MAXIMIZE)




        ### 9. Solve optimization problem
        # solve
        m.optimize()

        model_status = m.Status # if 2 a optimal solution was founded



        if model_status == 2:
            ######## create a dataframe with set as index
            solution = pd.DataFrame(index = index_bleaching)

            ######################## save optimal values - features of models (only the features) ########################

            # model d0eop_microkappa
            solution["especifico_dioxido_d0"] = especifico_dioxido_d0.gppd.X
            solution["especifico_oxigeno_eop"] = especifico_oxigeno_eop.gppd.X
            solution["especifico_peroxido_eop"] = especifico_peroxido_eop.gppd.X
            solution["especifico_soda_eop"] = especifico_soda_eop.gppd.X

            # model d1_brillo
            solution["especifico_dioxido_d1"] = especifico_dioxido_d1.gppd.X
            solution["especifico_acido_d1"] = especifico_acido_d1.gppd.X

            # model p_blancura
            solution["especifico_soda_p"] = especifico_soda_p.gppd.X
            solution["especifico_peroxido_p"] = especifico_peroxido_p.gppd.X
            solution["especifico_acido_p"] = especifico_acido_p.gppd.X


            ######################## save optimal values - targets of models (some targets are features of the model of the next step) ########################
            solution["microkappa_d1"] = microkappa_d1.gppd.X  # model d0eop_microkappa
            solution["brillo_entrada_p"] = brillo_entrada_p.gppd.X  # model d1_brillo
            solution["blancura_salida_p"] = blancura_salida_p.gppd.X  # model p_blancura


            ######################## round values ########################
            solution = solution.round(3)


            ######################## # get value objetive function ########################
            opt_cost = m.ObjVal




            ######## create a dataframe with set as index
            actual_values = pd.DataFrame(index = index_bleaching)

            ######################## save optimal values - features of models (only the features) ########################

            # model d0eop_microkappa
            actual_values["especifico_dioxido_d0"] = actual_value_especifico_dioxido_d0
            actual_values["especifico_oxigeno_eop"] = actual_value_especifico_oxigeno_eop
            actual_values["especifico_peroxido_eop"] = actual_value_especifico_peroxido_eop
            actual_values["especifico_soda_eop"] = actual_value_especifico_soda_eop

            # model d1_brillo
            actual_values["especifico_dioxido_d1"] = actual_value_especifico_dioxido_d1
            actual_values["especifico_acido_d1"] = actual_value_especifico_acido_d1

            # model p_blancura
            actual_values["especifico_soda_p"] = actual_value_especifico_soda_p
            actual_values["especifico_peroxido_p"] = actual_value_especifico_peroxido_p
            actual_values["especifico_acido_p"] = actual_value_especifico_acido_p
        else:
            pass






    ######################## INFO IN MAIN PAGE ########################
    

    # two tabs - first one show results - second show detail of model
    tab1, tab2 = st.tabs(["Results Optimization", "Details Optimization Modeling"])

    #### COLUMN1

    tab1.markdown("### ----- RESULTS OPTIMIZATION -----")
    if submitted_opt:
        if model_status == 2: # optimal solution was founded
            # show solution
            tab1.write(f"\n The optimal cost: ${opt_cost}")
            
            tab1.write('SOLUTION')
            tab1.dataframe(solution)

            tab1.write('ACTUAL VALUES')
            tab1.dataframe(actual_values)

            # download solution
            csv_solution = solution.to_csv()
            tab1.download_button("Download Optimal Solution", csv_solution, file_name='solution.csv', key='csv_key')
            #os.remove('solution.csv')
        else:
            tab1.write("Model is infeasible or unbounded - Change the input parameters")
    else:
        pass

    #### COLUMN2
    tab2.markdown("### ----- INFO OPTIMIZATION PROBLEM -----")
    tab2.write(" to do ")















