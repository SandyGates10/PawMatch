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

recomendados = df[df["cluster_gmm"] == cluster_user].copy()

# Guardamos copia del cluster sin filtros
recomendados_cluster = recomendados.copy()

# ============================
# FILTRO 1: Edad (PRIORIDAD)
# ============================

if edad == "cachorro":
    recomendados = recomendados[recomendados["ageGroup_baby"] == 1]
elif edad == "joven":
    recomendados = recomendados[recomendados["ageGroup_young"] == 1]
elif edad == "adulto":
    recomendados = recomendados[recomendados["ageGroup_adult"] == 1]
elif edad == "senior":
    recomendados = recomendados[recomendados["ageGroup_senior"] == 1]

# Si no hay perros de esa edad → relajamos edad pero avisamos
if len(recomendados) == 0:
    st.warning(
        "No encontramos perritos de esa edad en este grupo. "
        "Aquí tienes los más cercanos."
    )
    recomendados = recomendados_cluster.copy()

# ============================
# FILTRO 2: Tamaño (SECUNDARIO)
# ============================

recomendados_tamano = recomendados[recomendados["sizeGroup_ord"] == map_tamano[tamano]]

# Si hay suficientes del tamaño deseado → usamos esos
if len(recomendados_tamano) > 0:
    recomendados = recomendados_tamano
else:
    st.info(
        "No encontramos perritos de ese tamaño en este grupo, "
        "pero aquí tienes opciones de la edad que elegiste."
    )

st.subheader("🐶 Lomitos recomendados para ti")