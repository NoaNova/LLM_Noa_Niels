"""
PARTIE 7 - BOSS FINAL : Évaluation End-to-End du ChefBot
========================================================
Prérequis:
    pip install 'smolagents[litellm]' langfuse python-dotenv groq
    pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp

Ce script :
1. Crée un dataset de scénarios culinaires complexes.
2. Définit un Juge LLM expert en gastronomie.
3. Lance une comparaison entre deux modèles (ex: Llama-3.3-70B vs Llama-3.1-8B).
"""

import os
import json
import litellm
from datetime import datetime
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, tool
from langfuse import observe, get_client, Evaluation
from groq import Groq

load_dotenv()

# --- Configuration du Tracing Langfuse ---
litellm.callbacks = ["langfuse_otel"]

# Client pour le Juge
groq_client = Groq()

# =============================================================================
# OUTILS DU CHEF (Simulation du système pour l'évaluation)
# =============================================================================

@tool
def get_ingredient_prices(ingredients: str) -> str:
    """
    Récupère les prix moyens des ingrédients pour estimer le budget.
    Args:
        ingredients: Liste des ingrédients séparés par des virgules.
    """
    # Simulation simple de prix
    return "Prix moyens estimés: Viande (20€/kg), Légumes (4€/kg), Poisson (25€/kg), Épices (variable)."

@tool
def search_recipes(criteria: str) -> str:
    """
    Cherche des idées de recettes basées sur des critères (diète, type de plat).
    Args:
        criteria: Les critères de recherche (ex: "entrée végétarienne sans gluten").
    """
    return f"Suggestions trouvées pour '{criteria}': Salade de quinoa, Soupe de courge, Risotto aux champignons."

# =============================================================================
# 7.1 - DATASET DE SCÉNARIOS
# =============================================================================

def create_chef_dataset():
    """Crée le dataset 'chefbot-multiagent-eval' avec 4 niveaux de difficulté."""
    
    dataset_name = "chefbot-multiagent-eval"
    client = get_client()
    
    # Vérifie si le dataset existe déjà pour éviter les doublons
    try:
        client.get_dataset(dataset_name)
        print(f"Le dataset '{dataset_name}' existe déjà.")
        return
    except:
        pass

    print(f"Création du dataset '{dataset_name}'...")
    client.create_dataset(
        name=dataset_name,
        description="Scénarios d'évaluation pour le ChefBot Multi-Agent",
        metadata={"domain": "culinary_planning"}
    )

    test_cases = [
        # Scénario 1 : Facile
        {
            "input": {"question": "Je veux un dîner romantique simple pour 2 personnes ce soir."},
            "expected_output": {
                "must_respect": ["2 personnes", "romantique"],
                "expected_services": ["Entrée", "Plat", "Dessert"],
                "max_budget": "Flexible"
            }
        },
        # Scénario 2 : Moyen
        {
            "input": {"question": "Repas de famille pour 4. Attention, mon fils est allergique aux fruits à coque (noix, noisettes, etc.). Budget max 80€."},
            "expected_output": {
                "must_respect": ["4 personnes", "SANS FRUITS A COQUE", "Budget < 80€"],
                "expected_services": ["Entrée", "Plat", "Dessert"],
                "max_budget": 80
            }
        },
        # Scénario 3 : Difficile
        {
            "input": {"question": "Dîner pour 6 amis. 2 sont végétariens, 1 est intolérant au gluten. On a un petit budget de 15€ par personne max."},
            "expected_output": {
                "must_respect": ["6 personnes", "Options Végétariennes", "Option Sans Gluten", "Budget Total < 90€"],
                "expected_services": ["Apéritif", "Entrée", "Plat", "Dessert"],
                "max_budget": 90
            }
        },
        # Scénario 4 : Extrême
        {
            "input": {"question": "Grand événement pour 12 personnes. Besoin d'un menu Halal pour tous. Une personne est diabétique (pas de sucre ajouté). C'est pour une levée de fonds, budget illimité mais doit être très chic."},
            "expected_output": {
                "must_respect": ["12 personnes", "100% Halal", "Adapté Diabétique", "Chic/Gastronomique"],
                "expected_services": ["Amuse-bouche", "Entrée", "Plat", "Fromage", "Dessert"],
                "max_budget": 1000
            }
        }
    ]

    for case in test_cases:
        client.create_dataset_item(
            dataset_name=dataset_name,
            input=case["input"],
            expected_output=case["expected_output"],
        )
    
    print(f"Dataset créé avec {len(test_cases)} scénarios.")

# =============================================================================
# CONSTRUCTION DE L'AGENT (Configurable)
# =============================================================================

def build_chef_agent(model_id: str):
    """Construit l'agent Chef avec un modèle spécifique pour la comparaison."""
    
    model = LiteLLMModel(model_id=model_id, api_key=os.environ.get("GROQ_API_KEY"))
    
    return CodeAgent(
        tools=[get_ingredient_prices, search_recipes],
        model=model,
        name="ChefBot_Manager",
        description="Orchestrateur de cuisine qui crée des menus adaptés.",
        instructions="""Tu es un Chef Exécutif de renommée mondiale.
        Ta mission : Créer des menus complets qui respectent STRICTEMENT les contraintes des clients (allergies, budget, préférences).
        1. Analyse la demande.
        2. Vérifie les contraintes spéciales (Allergies = Priorité Absolue).
        3. Propose un menu détaillé.
        4. Estime le coût global.
        Sois créatif mais réaliste.""",
        max_steps=5, # Limité pour éviter les boucles infinies lors du test
    )

# =============================================================================
# 7.2 - JUGE LLM (Critères Gastronomiques)
# =============================================================================

CHEF_JUDGE_PROMPT = """Tu es un Critique Gastronomique expert (Guide Michelin).
Tu évalues la réponse d'un Chef IA à une demande client.

Données :
- Demande Client
- Réponse de l'IA
- Contraintes attendues (Allergies, Budget, etc.)

Note chaque critère de 0.0 à 1.0 :

1. **respect_contraintes** : CRITIQUE. Les allergies et régimes sont-ils respectés ? (0 si échec allergie).
2. **completude** : Le nombre de services (Entrée/Plat/Dessert) est-il conforme ?
3. **budget** : Le budget estimé semble-t-il respecté ?
4. **coherence** : Le menu a-t-il du sens culinairement ? (Pas de poisson avec du chocolat).
5. **faisabilite** : Les plats sont-ils réalisables ?

Réponds UNIQUEMENT en JSON :
{
    "respect_contraintes": 0.0,
    "completude": 0.0,
    "budget": 0.0,
    "coherence": 0.0,
    "faisabilite": 0.0,
    "explanation": "Brève justification en français"
}"""

@observe(name="chef-judge", as_type="generation")
def judge_chef_response(question: str, response: str, expected: dict) -> dict:
    """Appelle le LLM Juge pour noter la réponse."""
    
    must_respect = expected.get("must_respect", [])
    
    try:
        result = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile", # Le juge doit être performant
            messages=[
                {"role": "system", "content": CHEF_JUDGE_PROMPT},
                {"role": "user", "content": (
                    f"Demande Client: {question}\n\n"
                    f"Réponse Chef IA: {response}\n\n"
                    f"DOIT RESPECTER: {', '.join(str(x) for x in must_respect)}"
                )}
            ],
            temperature=0.0,
            response_format={"type": "json_object"} 
        )
        return json.loads(result.choices[0].message.content)
    except Exception as e:
        print(f"Erreur du juge: {e}")
        return {"respect_contraintes": 0, "explanation": "Erreur Juge"}

# =============================================================================
# 7.3 - EXÉCUTION ET COMPARAISON (CORRIGÉ V2)
# =============================================================================

def run_experiment(model_id: str, experiment_suffix: str):
    """Lance l'évaluation complète sur un modèle donné."""
    
    print(f"\n--- Lancement Expérience : {experiment_suffix} ({model_id}) ---")
    
    client = get_client()
    dataset = client.get_dataset("chefbot-multiagent-eval")
    agent = build_chef_agent(model_id)

    # --- CORRECTION 1 : La tâche doit accepter l'argument 'item' ---
    def task(item):
        # Langfuse passe un objet 'DatasetItem' nommé 'item'
        # On extrait la question du dictionnaire 'input' de cet item
        question = item.input["question"]
        
        # On exécute l'agent
        return str(agent.run(question))

    # --- CORRECTION 2 : L'évaluateur doit gérer les arguments dynamiques ---
    def culinary_evaluator(input, output, expected_output):
        # Langfuse passe automatiquement input, output et expected_output
        
        scores = judge_chef_response(
            question=input["question"], # input est le dict complet
            response=output,
            expected=expected_output
        )
        
        print(f"  > Juge ({experiment_suffix}): Note Budget={scores.get('budget', 0)}/1.0")

        return [
            Evaluation(name="respect_contraintes", value=scores.get("respect_contraintes", 0)),
            Evaluation(name="completude", value=scores.get("completude", 0)),
            Evaluation(name="budget", value=scores.get("budget", 0)),
            Evaluation(name="coherence", value=scores.get("coherence", 0)),
            Evaluation(name="faisabilite", value=scores.get("faisabilite", 0)),
        ]

    # Lancement de l'expérience
    results = client.run_experiment(
        name=f"chef-eval-{experiment_suffix}",
        data=dataset.items,
        task=task,
        evaluators=[culinary_evaluator],
        description=f"Évaluation du ChefBot avec {model_id}",
        metadata={
            "agent_model": model_id,
            "judge_model": "llama-3.3-70b-versatile",
        },
    )
    
    print(f"Expérience '{experiment_suffix}' terminée.")
    return results

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Étape 1 : Création du Dataset (Une seule fois suffit)
    create_chef_dataset()

    # Étape 2 : Comparaison de 2 Configurations
    
    # Config A : Modèle Puissant (Llama 3.3 70B)
    run_experiment(
        model_id="groq/llama-3.3-70b-versatile", 
        experiment_suffix="Model-70B"
    )

    # Config B : Modèle Rapide/Léger (Llama 3.1 8B)
    # On veut voir s'il gère bien les contraintes complexes (allergies)
    run_experiment(
        model_id="groq/llama-3.1-8b-instant", 
        experiment_suffix="Model-8B"
    )
    
    # Pour finir, on force l'envoi des traces
    get_client().flush()
    print("TOUTES LES ÉVALUATIONS SONT TERMINÉES.")
    print("Allez sur votre Dashboard Langfuse pour comparer les scores 'Model-70B' vs 'Model-8B'.")