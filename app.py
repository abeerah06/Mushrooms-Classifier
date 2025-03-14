import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import streamlit as st
user_input = pd.DataFrame()
df = pd.read_csv('mushrooms.csv')
X= df.drop(columns=['class','veil-type','veil-color','gill-attachment','stalk-shape'],axis=1)
y= df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
def prior_prob(y_train):
    counts = y_train.value_counts().to_dict()
    total = len(y_train)
    for i in counts:
        counts[i]/=total
    return counts
prior = prior_prob(y_train)
for i in prior:
    print(f'{i}:{prior[i]}')
def likehoodprob(X_train,y_train):
    class_counts = {
    }
    for i in X_train.columns:
        class_counts[i]={}
        for j in X_train[i].unique():
            class_counts[i][j]={}
            for k in y_train.unique():
                count = sum((X_train[i]==j)& (y_train== k))
                class_counts[i][j][k]=count
    return class_counts
likehood = likehoodprob(X_train,y_train)
def total_likehood(likehood):
    total_count = {}
    for i in likehood:
        total_count[i]={}
        for j in likehood[i]:
            total_count[i][j]=sum(likehood[i][j].values())
    return total_count

total = total_likehood(likehood)
def predict_naive_bayes(X_new, priors, likelihood_table):
    predictions = []
    
    for _, sample in X_new.iterrows():
        posteriors = {}
        
        for cls in priors.keys():
            posterior = priors[cls]
            
            for feature, value in sample.items():
                if value in likelihood_table[feature]:
                    posterior *= likelihood_table[feature][value].get(cls, 1.0)
            
            posteriors[cls] = posterior
        
        predictions.append(max(posteriors.items(), key=lambda x: x[1])[0])
    
    return predictions
st.markdown('''
    <style>
        [data-testid = 'stSidebar'] {
            background-color: #ffd6bd;
            color: #000000;
        }
        .st-emotion-cache-1104ytp h2 {
            font-size: 2.25rem;
            padding: 1rem 0px;
        }

    </style>
    ''', unsafe_allow_html=True
)
st.sidebar.image('7960wnzi8te11.webp')
st.sidebar.title('Mushroom Wizard')
st.sidebar.header("Terminology Guide")
with st.sidebar.expander("Cap Shape"):
        st.write("The shape of the mushroom cap. Examples:")
        st.write("- `b` - Bell-shaped")
        st.write("- `c` - Conical")
        st.write("- `x` - Convex")
        st.write("- `f` - Flat")
        st.write("- `k` - Knobbed")
        st.write("- `s` - Sunken")

with st.sidebar.expander("Cap Surface"):
        st.write("The texture of the mushroom cap. Examples:")
        st.write("- `f` - Fibrous")
        st.write("- `g` - Grooves")
        st.write("- `y` - Scaly")
        st.write("- `s` - Smooth")

with st.sidebar.expander("Cap Color"):
        st.write("The color of the mushroom cap. Examples:")
        st.write("- `n` - Brown")
        st.write("- `b` - Buff")
        st.write("- `c` - Cinnamon")
        st.write("- `g` - Gray")
        st.write("- `r` - Green")
        st.write("- `p` - Pink")
        st.write("- `u` - Purple")
        st.write("- `e` - Red")
        st.write("- `w` - White")
        st.write("- `y` - Yellow")

with st.sidebar.expander("Bruises"):
        st.write("Whether the mushroom has bruises (`t` - yes, `f` - no).")

with st.sidebar.expander("Odor"):
        st.write("The smell of the mushroom. Examples:")
        st.write("- `a` - Almond")
        st.write("- `l` - Anise")
        st.write("- `c` - Creosote")
        st.write("- `y` - Fishy")
        st.write("- `f` - Foul")
        st.write("- `m` - Musty")
        st.write("- `n` - None")
        st.write("- `p` - Pungent")
        st.write("- `s` - Spicy")

with st.sidebar.expander("Gill Spacing"):
        st.write("The spacing between gills. Examples:")
        st.write("- `c` - Close")
        st.write("- `w` - Crowded")
        st.write("- `d` - Distant")

with st.sidebar.expander("Gill Size"):
        st.write("The size of the gills. Examples:")
        st.write("- `b` - Broad")
        st.write("- `n` - Narrow")

with st.sidebar.expander("Gill Color"):
        st.write("The color of the gills. Examples:")
        st.write("- `k` - Black")
        st.write("- `n` - Brown")
        st.write("- `b` - Buff")
        st.write("- `h` - Chocolate")
        st.write("- `g` - Gray")
        st.write("- `r` - Green")
        st.write("- `o` - Orange")
        st.write("- `p` - Pink")
        st.write("- `u` - Purple")
        st.write("- `e` - Red")
        st.write("- `w` - White")
        st.write("- `y` - Yellow")

with st.sidebar.expander("Stalk Root"):
        st.write("The type of root attachment. Examples:")
        st.write("- `b` - Bulbous")
        st.write("- `c` - Club")
        st.write("- `u` - Cup")
        st.write("- `e` - Equal")
        st.write("- `z` - Rhizomorphs")
        st.write("- `r` - Rooted")
        st.write("- `?` - Missing data")

with st.sidebar.expander("Ring Number"):
        st.write("The number of rings on the mushroom stalk:")
        st.write("- `n` - None")
        st.write("- `o` - One")
        st.write("- `t` - Two")

with st.sidebar.expander("Ring Type"):
        st.write("The type of ring on the stalk:")
        st.write("- `c` - Cobwebby")
        st.write("- `e` - Evanescent")
        st.write("- `f` - Flaring")
        st.write("- `l` - Large")
        st.write("- `n` - None")
        st.write("- `p` - Pendant")
        st.write("- `s` - Sheathing")
        st.write("- `z` - Zone")

with st.sidebar.expander("Spore Print Color"):
        st.write("The color of the spore print:")
        st.write("- `k` - Black")
        st.write("- `n` - Brown")
        st.write("- `b` - Buff")
        st.write("- `h` - Chocolate")
        st.write("- `r` - Green")
        st.write("- `o` - Orange")
        st.write("- `u` - Purple")
        st.write("- `w` - White")
        st.write("- `y` - Yellow")

with st.sidebar.expander("Population"):
        st.write("How common the mushroom is in nature:")
        st.write("- `a` - Abundant")
        st.write("- `c` - Clustered")
        st.write("- `n` - Numerous")
        st.write("- `s` - Scattered")
        st.write("- `v` - Several")
        st.write("- `y` - Solitary")

with st.sidebar.expander("Habitat"):
        st.write("The environment where the mushroom grows:")
        st.write("- `g` - Grasses")
        st.write("- `l` - Leaves")
        st.write("- `m` - Meadows")
        st.write("- `p` - Paths")
        st.write("- `u` - Urban")
        st.write("- `w` - Waste")
        st.write("- `d` - Woods")
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if st.session_state.page == 'home':

    col1, col2 = st.columns(2)
    with col1:
        st.title('Mushroom Classifier')
    with col2:
        st.image('pngwing.com.png')
    st.write('This is a simple dashboard to classify mushrooms into edible or poisonous')
    col3, col4 = st.columns(2)
    with col3:
        cap_shape = st.selectbox("Cap Shape", ["x", "b", "s", "f", "p", "c"])
    with col4:
        cap_surface = st.selectbox("Cap Surface", ["s", "y", "f", "g"])
    col5, col6 = st.columns(2)
    with col5:
        cap_color = st.selectbox("Cap Color", ["n", "y", "w", "g"])
    with col6:
        bruises = st.selectbox("Bruises", ["t", "f"])
    col7, col8 = st.columns(2)
    with col7:
        odor = st.selectbox("Odor", ["a", "l", "c", "y", "f", "m", "n", "p", "s"])
    with col8:
        gill_spacing = st.selectbox("Gill Spacing", ["c", "w"])
    col9, col10 = st.columns(2)
    with col9:
        gill_size = st.selectbox("Gill Size", ["b", "n"])
    with col10:
        gill_color = st.selectbox("Gill Color", ["n", "b", "g", "r", "o", "p", "u", "e", "w", "y", "k", "h"])
    col11, col12 = st.columns(2)
    col13, col14 = st.columns(2)
    col15, col16 = st.columns(2)
    col17, col18 = st.columns(2)
    col19, col20 = st.columns(2)
    with col11:
        stalk_root = st.selectbox("Stalk Root", ["b", "c", "u", "e", "z", "r", "?"])
    with col12:
        stalk_surface_above_ring = st.selectbox("Stalk Surface Above Ring", ["s", "f", "y", "k"])
    with col13:
        stalk_surface_below_ring = st.selectbox("Stalk Surface Below Ring", ["s", "f", "y", "k"])
    with col14:
        stalk_color_above_ring = st.selectbox("Stalk Color Above Ring", ["n", "b", "c", "g", "o", "p", "e", "w", "y"])
    with col15:
        stalk_color_below_ring = st.selectbox("Stalk Color Below Ring", ["n", "b", "c", "g", "o", "p", "e", "w", "y"])
    with col16:
        ring_number = st.selectbox("Ring Number", ["n", "o", "t"])
    with col17:
        ring_type = st.selectbox("Ring Type", ["c", "e", "f", "l", "n", "p", "s", "z"])
    with col18:
        spore_print_color = st.selectbox("Spore Print Color", ["k", "n", "b", "h", "r", "o", "u", "w", "y"])
    with col19:
        population = st.selectbox("Population", ["a", "c", "n", "s", "v", "y"])
    with col20:
        habitat = st.selectbox("Habitat", ["g", "l", "m", "p", "u", "w", "d"])        
    user_input = {
        "cap-shape": cap_shape,
        "cap-surface": cap_surface,
        "cap-color": cap_color,
        "bruises": bruises,
        "odor": odor,
        "gill-spacing": gill_spacing,
        "gill-size": gill_size,
        "gill-color": gill_color,
        "stalk-root": stalk_root,
        "stalk-surface-above-ring": stalk_surface_above_ring,
        "stalk-surface-below-ring": stalk_surface_below_ring,
        "stalk-color-above-ring": stalk_color_above_ring,
        "stalk-color-below-ring": stalk_color_below_ring,
        "ring-number": ring_number,
        "ring-type": ring_type,
        "spore-print-color": spore_print_color,
        "population": population,
        "habitat": habitat
    }
    user_input = pd.DataFrame(user_input, index=[0])
    if st.button("Classify"):
        st.session_state.page = 'result'
        st.rerun()
elif st.session_state.page == 'result':
    prediction = predict_naive_bayes(user_input, prior, likehood)
    if prediction == 'e':
        prediction = 'Edible'
    else:
        prediction = 'Poisonous'
    st.title('Classification Results: ') 
    st.header(f'The mushroom is {prediction}')
    if st.button('Go Back'):
        st.session_state.page = 'home'
        st.rerun()