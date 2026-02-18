import streamlit as st
import pandas as pd
import numpy as np
import joblib
# rebuild

# ============================
# 1. Cargar modelo y datos
# ============================

@st.cache_resource
def load_model():
    return joblib.load("gmm_pawmatch.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("Paw_clusterizados.csv",encoding='latin1')

gmm = load_model()
df = load_data()

# ============================
# 2. Interfaz
# ============================

st.title("🐾 PawMatch – Encuentra a tu compañero ideal")

st.write(
    "Cuéntame qué buscas y te recomendaré perritos compatibles con la mascota "
    "que mejor se adapte a ti."
)

# ----------------------------
# Inputs simples del usuario
# ----------------------------

tamano = st.selectbox("Tamaño preferido", ["pequeño", "mediano", "grande"])
energia = st.selectbox("Nivel de energía", ["bajo", "medio", "alto"])
edad = st.selectbox("Edad preferida", ["cachorro", "joven", "adulto", "senior"])

compat_perros = st.checkbox("Tengo otros perros")
compat_gatos = st.checkbox("Tengo algún gato")
compat_ninos = st.checkbox("Tengo niños")

actividad = st.selectbox("Tu nivel de actividad diario", ["tranquilo", "moderado", "activo"])

# ============================
# 3. Mapear inputs a variables internas
# ============================

map_tamano = {"pequeño": 0, "mediano": 1, "grande": 2}
map_energia = {"bajo": 0, "medio": 1, "alto": 2}
map_actividad = {"tranquilo": 0, "moderado": 1, "activo": 2}

# Dummies de edad
age_adult = 1 if edad == "adulto" else 0
age_baby = 1 if edad == "cachorro" else 0
age_senior = 1 if edad == "senior" else 0
age_young = 1 if edad == "joven" else 0

# ============================
# 4. Construir vector completo (alineado al GMM)
# ============================

X_user = pd.DataFrame([{
    "adoptionFee": 0,
    "num_breeds": 1,
    "compat_score": 0,
    "care_score": 0,
    "isCourtesyListing": 0,
    "isNeedingFoster": 0,
    "isSponsorable": 0,
    "isCatsOk": int(compat_gatos),
    "isDogsOk": int(compat_perros),
    "isKidsOk": int(compat_ninos),
    "isSpecialNeeds": 0,
    "isHousetrained": 0,
    "coatLength_ord": 1,
    "activityLevel_ord": map_actividad[actividad],
    "energyLevel_ord": map_energia[energia],
    "sizeGroup_ord": map_tamano[tamano],
    "obedienceTraining_ord": 1,
    "ageGroup_adult": age_adult,
    "ageGroup_baby": age_baby,
    "ageGroup_senior": age_senior,
    "ageGroup_young": age_young,
    "sex_female": 0,
    "sex_male": 0,
    "newPeopleReaction_aggressive": 0,
    "newPeopleReaction_cautious": 0,
    "newPeopleReaction_friendly": 1,
    "newPeopleReaction_protective": 0
}])

# ============================
# 5. Predicción del cluster
# ============================

cluster_user = gmm.predict(X_user)[0]

st.subheader(f"Tu mascota ideal pertenece al cluster: **{cluster_user}**")

# ============================
# 6. Recomendaciones
# ============================

recomendados = df[df["cluster_gmm"] == cluster_user].copy()

st.subheader("🐶 Lomitos recomendados para ti")

# Mapeos bonitos para mostrar al usuario
map_tamano_rev = {0: "Pequeño", 1: "Mediano", 2: "Grande"}
map_energia_rev = {0: "Baja", 1: "Media", 2: "Alta"}

# Estilos para las tarjetas tipo Tinder (se inyectan una sola vez)
st.markdown(
    """
    <style>
    .card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .name {
        font-size: 24px;
        font-weight: bold;
        color: #333333;
    }
    .info {
        font-size: 16px;
        color: #555555;
    }
    </style>
    """,
    unsafe_allow_html=True
)

for _, row in recomendados..sample(10).iterrows():

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])

        # Foto
        with col1:
            if "pictureThumbnailUrl" in row and pd.notna(row["pictureThumbnailUrl"]):
                st.image(row["pictureThumbnailUrl"], width=200)
            else:
                st.write("Sin foto disponible")

        # Información
        with col2:
            nombre = row["name"] if "name" in row else "Perrito"

            edad_texto = (
                "Adulto" if row["ageGroup_adult"] else
                "Cachorro" if row["ageGroup_baby"] else
                "Joven" if row["ageGroup_young"] else
                "Senior" if row["ageGroup_senior"] else
                "Sin dato"
            )

            st.markdown(f'<div class="name">{nombre}</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="info">
                🐾 <b>Edad:</b> {edad_texto}<br>
                📏 <b>Tamaño:</b> {map_tamano_rev.get(row['sizeGroup_ord'], 'N/A')}<br>
                ⚡ <b>Energía:</b> {map_energia_rev.get(row['energyLevel_ord'], 'N/A')}<br>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)