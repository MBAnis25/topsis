import streamlit as st
import numpy as np
import pandas as pd
import io  

# Define the functions used in the script

def assign_zones_and_actions(data):
    data["Prod Oil"] = pd.to_numeric(data["Prod Oil"], errors='coerce')
    data["Prod Water"] = pd.to_numeric(data["Prod Water"], errors='coerce')
    conditions = [
        (data["Prod Oil"] > 0) & (data["Prod Water"] > 0),
        (data["Prod Oil"] < 0) & (data["Prod Water"] < 0),
        (data["Prod Oil"] < 0) & (data["Prod Water"] > 0)
    ]
    choices = ["1 : High Oil & High Water", "2 : Low Oil & Low Water", "3 : Low Oil & High Water"]
    data['Zone'] = np.select(conditions, choices, default='Zone Unknown')
    return data

def calculer_vecteur_priorite(matrice):
    somme_colonnes = np.sum(matrice, axis=0)
    matrice_normalisee = matrice / somme_colonnes
    vecteur_priorite = np.mean(matrice_normalisee, axis=1)
    vecteur_priorite_arrondi = np.round(vecteur_priorite, 3)
    return vecteur_priorite_arrondi

def calculate_normalized_weighted_values(data, weights):
    weighted_normalized_data = (data / (np.sqrt(np.sum(data**2, axis=0)))) * weights
    return weighted_normalized_data

def calculate_ideal_and_anti_ideal_points(weighted_data, criteria_types):
    ideal = []
    anti_ideal = []
    for idx, maximize in enumerate(criteria_types):
        if maximize:
            ideal.append(np.max(weighted_data[:, idx]))
            anti_ideal.append(np.min(weighted_data[:, idx]))
        else:
            ideal.append(np.min(weighted_data[:, idx]))
            anti_ideal.append(np.max(weighted_data[:, idx]))
    return np.array(ideal), np.array(anti_ideal)

def calculate_distances_and_scores(weighted_data, ideal_point, anti_ideal_point):
    distances_to_ideal = np.sqrt(np.sum((weighted_data - ideal_point)**2, axis=1))
    distances_to_anti_ideal = np.sqrt(np.sum((weighted_data - anti_ideal_point)**2, axis=1))
    
    # Ensure the denominator is not zero before dividing
    # Adding a small epsilon value to the denominator ensures there is no division by zero
    epsilon = 1e-10  # A small value close to zero
    scores = distances_to_anti_ideal / (distances_to_ideal + distances_to_anti_ideal + epsilon)
    scores = np.round(scores, 5)
    
    return distances_to_ideal, distances_to_anti_ideal, scores


def select_best_action(scores):
    best_index = np.argmax(scores)
    return best_index + 1, scores[best_index]

def rank_actions(scores):
    ranked_indices = np.argsort(scores)[::-1] + 1
    return ranked_indices, scores[ranked_indices - 1]


def verifier_matrice(matrice):
    n = matrice.shape[0]
    # Vérifier si la diagonale contient tous des 1
    if not np.all(np.diag(matrice) == 1):
        return False, "La diagonale de la matrice doit contenir uniquement des 1."

    # Vérifier si les éléments au-dessus de la diagonale sont les inverses de ceux en dessous
    for i in range(n):
        for j in range(i+1, n):
            if matrice[i, j] != 1 / matrice[j, i]:
                return False, f"Élément [{i},{j}] n'est pas l'inverse de l'élément [{j},{i}]."
    return True, "La matrice est valide."



# Define available zones
zones_disponibles = {
    "1": "1 : High Oil & High Water",  # High Oil & High Water
    "2": "2 : Low Oil & Low Water",  # Low Oil & Low Water
    "3": "3 : Low Oil & High Water"   # Low Oil & High Water
}






# Define the functions used in the script
# All previous functions are the same

def main():
    st.title('TOPSIS Method for Well Ranking')

    with st.sidebar:
        st.header("Inputs")
        uploaded_file = st.file_uploader("Upload your Excel file", key="file")

        if uploaded_file is not None:
            try:
                # Load data from different sheets
                weight_slb = pd.read_excel(uploaded_file, sheet_name='CC', header=None)
                df = pd.read_excel(uploaded_file, sheet_name='AC', skipfooter=1)
                type1 = pd.read_excel(uploaded_file, sheet_name='AC')

                # Process data
                df = assign_zones_and_actions(df)

                # Verify weight matrix
                matrice_comparaison = weight_slb.iloc[1:, 1:].to_numpy(dtype=float)
                valid, message = verifier_matrice(matrice_comparaison)
                if not valid:
                    st.error(message)
                    return

                # Allow user to select a zone
                zone_option = st.selectbox('Choose a zone', list(zones_disponibles.values()))
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                return

    if uploaded_file is not None and 'zone_option' in locals():
        filtered_data = df[df["Zone"] == zone_option].copy()
        if filtered_data.empty:
            st.write(f"No data available for the selected zone: {zone_option}")
        else:
            actions = filtered_data['Actions']
            data = filtered_data.iloc[:, 1:-1].values.astype(float)
            criteria_types_row_index = type1.index[-1]
            criteria_types = type1.iloc[criteria_types_row_index, 1:6].apply(lambda x: x.lower() in ['true', 'max', 'maximize']).values

            # Calculations
            weights = calculer_vecteur_priorite(matrice_comparaison)
            weighted_normalized_data = calculate_normalized_weighted_values(data, weights)
            ideal_point, anti_ideal_point = calculate_ideal_and_anti_ideal_points(weighted_normalized_data, criteria_types)
            distances_to_ideal, distances_to_anti_ideal, scores = calculate_distances_and_scores(weighted_normalized_data, ideal_point, anti_ideal_point)

            # Rank actions based on scores
            ranked_indices = np.argsort(scores)
            ranked_scores = scores[ranked_indices]

            results_df = pd.DataFrame({
                "Action": actions.iloc[ranked_indices].values,
                "Score": ranked_scores,
                "Rank": np.arange(1, len(ranked_scores) + 1)
            })

            # Display results
            st.write(f"Final ranking for the zone {zone_option}")
            st.dataframe(results_df)

            # Create an Excel file in memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                results_df.to_excel(writer, index=False, sheet_name='Results')

            # Download link for the results DataFrame as Excel
            st.download_button(
                label="Download results as Excel",
                data=output.getvalue(),
                file_name='results.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

if __name__ == "__main__":
    main()