# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:41:48 2020

@author: KevinG
"""


import streamlit as st

st.title('Reinforcement Learning in Inventory Management')

unsatisfied_demand = st.selectbox(
    'What to do in case the demand can\'t be satsified?',
     ('backorder', 'lost sales'))

goal = st.selectbox(
    'What is the goal of the case?',
     ('minimize costs', 'target service level'))

order_policy = st.selectbox(
    'What is the order policy?',
     ('X+Y', '(s,S)'))

iterations = st.number_input('Number of iterations')

alpha = st.number_input('Learning rate')

n = st.number_input('horizon')

exploitation = st.slider('Exploitation rate', 0.0, 1.0)

no_suppliers = st.number_input('Number of suppliers',
                               min_value=1, value=1)

no_customers = st.number_input('Number of customers', min_value=1, value=1)

no_stockpoints = st.number_input('Number of stockpoints')
