import streamlit as st
import joblib
from naive import predict_naive_bayes
model_data = joblib.load("mushroom_classifier.pkl")
p = model_data["prior"]
l = model_data["likelihood"]
print(p)
st.markdown('''
    <style>
        [data-testid = 'stSidebar'] {
            background-color: #ffd6bd;
        }

    </style>
    ''', unsafe_allow_html=True
)
col1, col2 = st.columns(2)
with col1:
    st.title('Mushroom Classifier')
with col2:
    st.image('pngwing.com.png')
st.write('This is a simple dashboard to classify mushrooms into edible or poisonous')
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
cap_shape = st.selectbox("Cap Shape", ["x", "b", "s", "f", "p", "c"])
cap_surface = st.selectbox("Cap Surface", ["s", "y", "f", "g"])
cap_color = st.selectbox("Cap Color", ["n", "y", "w", "g"])
bruises = st.selectbox("Bruises", ["t", "f"])
odor = st.selectbox("Odor", ["a", "l", "c", "y", "f", "m", "n", "p", "s"])
gill_spacing = st.selectbox("Gill Spacing", ["c", "w"])
gill_size = st.selectbox("Gill Size", ["b", "n"])
gill_color = st.selectbox("Gill Color", ["n", "b", "g", "r", "o", "p", "u", "e", "w", "y", "k", "h"])
stalk_root = st.selectbox("Stalk Root", ["b", "c", "u", "e", "z", "r", "?"])
stalk_surface_above_ring = st.selectbox("Stalk Surface Above Ring", ["s", "f", "y", "k"])
stalk_surface_below_ring = st.selectbox("Stalk Surface Below Ring", ["s", "f", "y", "k"])
stalk_color_above_ring = st.selectbox("Stalk Color Above Ring", ["n", "b", "c", "g", "o", "p", "e", "w", "y"])
stalk_color_below_ring = st.selectbox("Stalk Color Below Ring", ["n", "b", "c", "g", "o", "p", "e", "w", "y"])
ring_number = st.selectbox("Ring Number", ["n", "o", "t"])
ring_type = st.selectbox("Ring Type", ["c", "e", "f", "l", "n", "p", "s", "z"])
spore_print_color = st.selectbox("Spore Print Color", ["k", "n", "b", "h", "r", "o", "u", "w", "y"])
population = st.selectbox("Population", ["a", "c", "n", "s", "v", "y"])
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
user_input1= list(user_input.values())
if st.button("Classify"):
    prediction = predict_naive_bayes(user_input1, p, l)
    st.write(f"The mushroom is {prediction}")