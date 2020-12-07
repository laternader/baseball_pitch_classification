import pickle
import streamlit as st

st.title('Strike or not?')

# Any model can go in here for right now
with open('assets/rf_no_pitcher12062020.pkl', 'rb') as pickle_in:
    model = pickle.load(pickle_in)

@st.cache
def load_data():
    df = pd.read_csv('data/app_train.py')
    return df

batter = st.sidebar.selectbox(
    'Select Batter',
    #    How do I put a list of batters here using df  ['batter_name'].unique()
)

@st.cache
def 

# 

