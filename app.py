import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================
# 1. Cargar modelo y datos
# ============================

@st.cache_resource
def load_model():
    return joblib.load("pawmatch.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("Paw_clusterizados.csv", encoding='latin1')

gmm = load_model()
df = load_data()

# Lista EXACTA de columnas usadas para entrenar el GMM
features = [
    "adoptionFee", "num_breeds", "compat_score", "care_score",
    "isCourtesyListing", "isNeedingFoster", "isSponsorable",
    "isCatsOk", "isDogsOk", "isKidsOk", "isSpecialNeeds",
    "isHousetrained", "coatLength_ord", "activityLevel_ord",
    "energyLevel_ord", "sizeGroup_ord", "obedienceTraining_ord",
    "ageGroup_adult", "ageGroup_baby", "ageGroup_senior",
    "ageGroup_young", "sex_female", "sex_male",
    "newPeopleReaction_cautious", "newPeopleReaction_friendly"
]

# ============================
# 2. Interfaz
# ============================

st.title("🐾 PawMatch – Encuentra a tu compañero ideal")

st.write(
    "Cuéntame qué buscas y te recomendaré perritos compatibles "
    "con la mascota que mejor se adapte a ti."
)

# ----------------------------
# Inputs del usuario
# ----------------------------

tamano = st.selectbox("Tamaño preferido", ["pequeño", "mediano", "grande"])
edad = st.selectbox("Edad preferida", ["cachorro", "joven", "adulto", "senior"])

compat_perros = st.checkbox("Tengo otros perros")
compat_gatos = st.checkbox("Tengo algún gato")
compat_ninos = st.checkbox("Tengo niños")

actividad = st.selectbox("Tu nivel de actividad diario", ["tranquilo", "moderado", "activo"])

# NUEVA PREGUNTA
contacto_personas = st.selectbox(
    "¿Tu lomito estará en constante contacto con más personas?",
    ["sí", "no"]
)

# ============================
# 3. Mapear inputs
# ============================

map_tamano = {"pequeño": 0, "mediano": 1, "grande": 2}
map_actividad = {"tranquilo": 0, "moderado": 1, "activo": 2}

# Dummies de edad
age_adult = 1 if edad == "adulto" else 0
age_baby = 1 if edad == "cachorro" else 0
age_senior = 1 if edad == "senior" else 0
age_young = 1 if edad == "joven" else 0

# Sociabilidad
friendly = 1 if contacto_personas == "sí" else 0
cautious = 1 - friendly

# ============================
# 4. Construir X_user refinado
# ============================

prom = df.mean(numeric_only=True)

X_user = pd.DataFrame([{
    # Variables del usuario
    "isCatsOk": int(compat_gatos),
    "isDogsOk": int(compat_perros),
    "isKidsOk": int(compat_ninos),
    "activityLevel_ord": map_actividad[actividad],
    "sizeGroup_ord": map_tamano[tamano],
    "ageGroup_adult": age_adult,
    "ageGroup_baby": age_baby,
    "ageGroup_senior": age_senior,
    "ageGroup_young": age_young,
    "newPeopleReaction_friendly": friendly,
    "newPeopleReaction_cautious": cautious,

    # Variables no controladas → promedio del dataset
    "adoptionFee": prom["adoptionFee"],
    "num_breeds": prom["num_breeds"],
    "compat_score": prom["compat_score"],
    "care_score": prom["care_score"],
    "isCourtesyListing": 0,
    "isNeedingFoster": 0,
    "isSponsorable": 0,
    "isSpecialNeeds": 0,
    "isHousetrained": prom["isHousetrained"],
    "coatLength_ord": prom["coatLength_ord"],
    "obedienceTraining_ord": prom["obedienceTraining_ord"],
    "energyLevel_ord": prom["energyLevel_ord"],  # ← PROMEDIO EN VEZ DE PREGUNTA
    "sex_female": 0,
    "sex_male": 0,
}])

# Asegurar orden correcto
X_user = X_user[features]

# ============================
# 5. Predicción del cluster
# ============================

cluster_names = {
    0: "Sociables y tranquilos",
    1: "Jóvenes amables",
    2: "Pequeños y bonitos",
    3: "Los más cariñosos",
    4: "Peluditos sensibles"
}

cluster_user = gmm.predict(X_user)[0]
cluster_name = cluster_names.get(cluster_user, "Cluster desconocido")

st.subheader(f"Tu mascota ideal pertenece al grupo: **{cluster_name}**")

# ============================
# 6. Recomendaciones
# ============================

# 1. Orden de clusters: primero el predicho, luego los demás
clusters_en_orden = [cluster_user] + [c for c in df["cluster_gmm"].unique() if c != cluster_user]

recomendados = None
cluster_final = None

for c in clusters_en_orden:
    candidatos = df[df["cluster_gmm"] == c].copy()

    # FILTRO 1: Edad
    if edad == "cachorro":
        candidatos = candidatos[candidatos["ageGroup_baby"] == 1]
    elif edad == "joven":
        candidatos = candidatos[candidatos["ageGroup_young"] == 1]
    elif edad == "adulto":
        candidatos = candidatos[candidatos["ageGroup_adult"] == 1]
    elif edad == "senior":
        candidatos = candidatos[candidatos["ageGroup_senior"] == 1]

    # Si no hay nada de esa edad en este cluster → probar siguiente cluster
    if len(candidatos) == 0:
        continue

    # FILTRO 2: Tamaño
    candidatos_tamano = candidatos[candidatos["sizeGroup_ord"] == map_tamano[tamano]]

    # Si hay tamaño + edad → perfecto
    if len(candidatos_tamano) > 0:
        recomendados = candidatos_tamano
        cluster_final = c
        break

    # Si no hay tamaño pero sí edad → usar edad
    recomendados = candidatos
    cluster_final = c
    break

# Si después de revisar todos los clusters no hay nada → usar TODO el dataset
if recomendados is None or len(recomendados) == 0:
    recomendados = df.copy()
    cluster_final = None

# Nombre del cluster final
if cluster_final is not None:
    cluster_name = cluster_names.get(cluster_final, "Grupo desconocido")
else:
    cluster_name = "Opciones generales del refugio"

st.subheader(f"Tu mascota ideal pertenece al grupo: **{cluster_name}**")

# ============================
# Mostrar tarjetas (TOP 5)
# ============================

st.subheader("🐶 Lomitos recomendados para ti")

map_tamano_rev = {0: "Pequeño", 1: "Mediano", 2: "Grande"}
map_energia_rev = {0: "Baja", 1: "Media", 2: "Alta"}

# Estilos
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

# TOP 5 garantizado
top5 = recomendados.head(5)

for _, row in top5.iterrows():

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
            nombre = row.get("name", "Perrito")

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