import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('big5.csv')

# Encode categorical variables
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Personality (Class label)'] = label_encoder.fit_transform(df['Personality (Class label)'])

# Display title and subtitle
st.write('''
# Big Five
Ketahui tipe kepribadianmu berdasarkan parameter di bawah ini!
''')

# Display information about data
st.subheader('Informasi Data:')
st.dataframe(df)
st.write(df.describe())
st.write(df.dtypes)

st.subheader('Distribusi Faktor Kepribadian:')
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

traits = ['openness', 'neuroticism', 'conscientiousness', 'agreeableness', 'extraversion']
for i, trait in enumerate(traits):
    row, col = divmod(i, 3)
    sns.histplot(df[trait], kde=True, ax=axes[row, col])
    axes[row, col].set_title(trait)

st.pyplot(fig)

# Split data into X and Y
X = df.iloc[:, 0:7].values
Y = df.iloc[:, -1].values

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Get user input
def get_user_input():
    Gender = st.selectbox("Masukkan Gender Anda", ["Female", "Male"])
    Age = st.slider('Usia', 5, 28, 5)
    openness = st.slider('Seberapa terbuka Anda terhadap ide dan pengalaman baru', 1, 8, 1)
    neuroticism = st.slider('Seberapa sensitif Anda terhadap stress dan pemicu emosi negatif', 1, 8, 1)
    conscientiousness = st.slider('Seberapa terarah pada tujuan, gigih, dan terorganisirnya Anda', 1, 8, 1)
    agreeableness = st.slider('Seberapa besar Anda mendahulukan kepentingan dan kebutuhan orang lain di atas kepentingannya sendiri', 1, 8, 1)
    extraversion = st.slider('Seberapa besar Anda diberi energi oleh dunia luar', 1, 8, 1)

    user_data = {
        'Gender': Gender,
        'Age': Age,
        'openness': openness,
        'neuroticism': neuroticism,
        'conscientiousness': conscientiousness,
        'agreeableness': agreeableness,
        'extraversion': extraversion
    }

    features = pd.DataFrame(user_data, index=[0])
    return features

def get_probabilities(prediction_probs):
    class_labels = ['DEPENDABLE', 'EXTRAVERTED', 'LIVELY', 'RESPONSIBLE', 'SERIOUS']
    probabilities = dict(zip(class_labels, prediction_probs))
    return probabilities

def Hasil(prediction_probs):
    probabilities = get_probabilities(prediction_probs)

    # Display probabilities
    st.subheader('Probabilitas untuk setiap Kelas Kepribadian:')
    for personality_class, probability in probabilities.items():
        st.write("{}: {:.2f}".format(personality_class, probability))

    # Determine the predicted personality class
    predicted_class = max(probabilities, key=probabilities.get)

    # Display the predicted personality class
    st.subheader('Kepribadian yang Diprediksi:')
    st.write(predicted_class)

    # Display additional information based on the predicted class
    st.write(get_personality_info(predicted_class))

    # Create a bar chart for visualization
    fig, ax = plt.subplots()
    ax.bar(probabilities.keys(), probabilities.values())
    ax.set_ylabel('Probabilitas')
    ax.set_title('Probabilitas untuk Setiap Kelas Kepribadian')
    st.pyplot(fig)

    return predicted_class

# ...

def get_personality_info(predicted_class):
    if predicted_class == 'SERIOUS':
        return "(SERIOUS) Orang dengan kepribadian ini cenderung bersikap sungguh-sungguh, fokus, dan serius dalam berbagai hal. Mereka mungkin menempatkan tanggung jawab sebagai prioritas utama dan menganggap komitmen dengan serius. Namun, penting untuk diingat bahwa kepribadian adalah sesuatu yang kompleks dan dapat bervariasi tergantung pada situasi tertentu. Jika ada konteks khusus atau informasi tambahan, saya dapat memberikan informasi yang lebih terarah."
    elif predicted_class == 'RESPONSIBLE':
        return "(RESPONSIBLE) Orang dengan kepribadian yang bertanggung jawab cenderung dapat diandalkan, menjalankan tugas dan kewajiban mereka dengan serius, dan memenuhi komitmen mereka. Mereka biasanya peduli terhadap konsekuensi dari tindakan mereka dan berusaha untuk melakukan yang terbaik dalam mengelola tanggung jawab mereka."
    elif predicted_class == 'LIVELY':
        return "(LIVELY) Orang dengan kepribadian yang hidup cenderung memiliki kegembiraan dan semangat yang menonjol dalam berbagai situasi. Mereka mungkin terlihat bersemangat, ramah, dan suka berinteraksi dengan orang lain."
    elif predicted_class == 'EXTRAVERTED':
        return "(EXTRAVERTED) Orang extrovert lebih menyukai lingkungan yang interaktif. Tipe kepribadian yang satu ini biasanya dimiliki oleh orang yang perhatiannya diarahkan ke luar dirinya sendiri."
    else:
        return "(DEPENDABLE) Dependable personality merujuk pada sifat atau ciri kepribadian seseorang yang dapat diandalkan, dapat dipercaya, dan dapat diandalkan untuk menepati janji atau tanggung jawabnya. Orang dengan kepribadian yang dapat diandalkan cenderung memenuhi komitmen mereka, bekerja secara konsisten, dan dapat diandalkan oleh orang lain."

# Display user input
user_input = get_user_input()
st.subheader('Input Pengguna:')
st.write(user_input)

# Create and train the model
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Display personality prediction
label_encoder = LabelEncoder()
user_input['Gender'] = label_encoder.fit_transform(user_input['Gender'])
prediction_probs = model.predict_proba(user_input)
st.subheader('Prediksi Kepribadian:')
st.write(Hasil(prediction_probs[0]))