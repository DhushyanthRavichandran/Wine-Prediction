import streamlit as st
import numpy as np
import pandas

import joblib
import pickle


model_filename = 'winequality.joblib'
model = joblib.load(model_filename)


st.title('Wine quality')
st.write('''Embark on a sensory journey with wine, a symphony of flavors crafted by nature and human expertise. Whether indulging in the bold richness of Cabernet Sauvignon, the crispness of Chardonnay, or the delicate charm of Pinot Noir, each glass tells a unique tale of terroir and tradition.

Wine isn't just a beverage; it's a cultural phenomenon, a timeless companion to rituals and shared moments. Uncork a bottle, and you're transported to sun-soaked vineyards, ancient cellars, and a legacy steeped in history.

Guided by sommeliers, the language of wine is poetic, weaving tasting notes that invite you to explore a world of aromas and textures. Pairing wine with food becomes an art, transforming meals into sensory delights and creating lasting memories.

Innovation meets tradition in the wine world, where old-world wineries and sustainable practices thrive. As you savor each sip, raise your glass to the enduring connection between humans and the Earth—a connection celebrated in every drop of this exquisite elixir. Cheers!''')

st.markdown("## Types of wine")

tab1, tab2= st.tabs(["## White wine", "## Red wine"])
st.divider()

with tab1:
   
   st.subheader("White wine",divider='gray',anchor=False)
   st.write('''
White wine, a diverse and enticing category, encompasses an array of flavors and aromas crafted from fermented grape juice without skins. Varied grape varietals like Chardonnay, Sauvignon Blanc, and Riesling contribute to the spectrum of tastes, from zesty citrus to tropical richness. Vinification methods, including stainless steel fermentation and oak aging, lend distinct characteristics, while styles range from still to sparkling, with Champagne and Prosecco among the effervescent highlights. White wine's versatility shines in food pairings, enhancing the dining experience with its compatibility with seafood, salads, and poultry. Served chilled, white wines showcase their refreshing nature, with ideal temperatures varying by varietal. Notable regions such as Burgundy, France, and New Zealand excel in producing Chardonnay and Sauvignon Blanc, respectively. While many white wines are enjoyed young, certain varietals, like aged Chardonnays, reveal added complexity over time. In essence, white wine invites exploration, offering a delightful journey through a world of flavors, textures, and cultural nuances.''')

with tab2:
   st.subheader("Red wine",divider='red',anchor=False)
   st.write('''
Red wine, a captivating elixir derived from fermented dark grape varieties, boasts a rich tapestry of flavors, ranging from bold and fruity to complex and robust. The primary grape types, including Cabernet Sauvignon, Merlot, and Pinot Noir, contribute distinctive profiles to the world of red wines. The winemaking process, involving maceration and fermentation with grape skins, imparts deep color, tannins, and a myriad of aromas. Red wines can be classified into Old World and New World styles, with regions like Bordeaux, France, and Napa Valley, USA, showcasing their unique terroir. Aging in oak barrels adds layers of complexity, introducing notes of vanilla, spice, and smokiness. Red wine pairs exquisitely with hearty dishes such as steak, pasta, and mature cheeses, enhancing the dining experience. Serving temperatures vary by varietal, with lighter reds benefiting from a slight chill, while full-bodied reds unfold their nuances at room temperature. Red wine also holds a cultural significance, symbolizing celebrations, rituals, and convivial gatherings. In essence, red wine is a timeless indulgence, inviting enthusiasts to savor the artistry of winemaking and the pleasures of the palate.''')


st.markdown(
        """<style>
    div[class*="stSlider"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 15px;
    }
        </style>

        """, unsafe_allow_html=True)



fixed_acidity=st.sidebar.slider('fixed acidity', min_value=3.8, max_value=15.8)

volatile_acidity=st.sidebar.slider('volatile acidity', min_value=0.08, max_value=1.5)

citric_acid=st.sidebar.slider('citric acid', min_value=0.0, max_value=1.6)

residual_sugar=st.sidebar.slider('residual sugar', min_value=0.6, max_value=65.0)

chlorides=st.sidebar.slider('chlorides', min_value=0.01, max_value=0.6)

free_sulfur_dioxide=st.sidebar.slider('free sulfur dioxide', min_value=0, max_value=280)

total_sulfur_dioxide=st.sidebar.slider('total sulfur dioxide', min_value=5, max_value=440)

density=st.sidebar.slider('density', min_value=1.0, max_value=1.04)

pH=st.sidebar.slider('pH', min_value=2.72, max_value=4.0)

sulphates=st.sidebar.slider('sulphates', min_value=0.2, max_value=2.0)

alcohol=st.sidebar.slider('alcohol', min_value=8.0, max_value=14.8)

quality=st.sidebar.slider('quality', min_value=3, max_value=9)

features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, quality]])

prediction = model.predict(features)



st.write('''
Explore our wine prediction platform featuring an advanced XGBoost model boasting a 95% accuracy in classifying wines as red or white. This powerful tool unveils intricate patterns, providing precise predictions for enthusiasts and industry professionals. Check out the predicted results to make informed decisions and elevate your wine experiences. Cheers to accurate insights!''')

# Provide instructions
st.write('## How to Use:')
st.write('Adjust the sliders to input different wine features, and the model will predict whether the wine is red or white.')


threshold = 0.6 # Threshold for model classification
predicted_wine_type = "Red Wine" if prediction[0] >= threshold else "White Wine"
st.write(f'## Predicted Wine Type: {predicted_wine_type}')

