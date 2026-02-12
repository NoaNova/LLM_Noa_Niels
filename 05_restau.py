import os
from dotenv import load_dotenv
from smolagents import CodeAgent, Tool, LiteLLMModel, tool
import datetime

# Chargement des variables d'environnement
load_dotenv()

# Configuration du modèle (Groq via LiteLLM pour smolagents)
# On utilise un modèle performant pour la planification
model = LiteLLMModel(
    model_id="groq/llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY")
)

# =============================================================================
# 5.1 - OUTIL BASE DE DONNEES (Héritage de la classe Tool)
# =============================================================================

class MenuDatabaseTool(Tool):
    def __init__(self):
        # Définition des métadonnées de l'outil pour le LLM
        self.name = "menu_database"
        self.description = "Recherche des plats dans le menu selon des critères (catégorie, prix max, allergènes à exclure)."
        self.inputs = {
            "category": {
                "type": "string", 
                "description": "La catégorie du plat (entrée, plat, dessert, boisson). Optionnel.",
                "nullable": True
            },
            "max_price": {
                "type": "number", 
                "description": "Prix maximum par plat. Optionnel.",
                "nullable": True
            },
            "exclude_allergen": {
                "type": "string", 
                "description": "Allergène à éviter (ex: 'gluten', 'fruits à coque', 'viande' pour végétarien). Optionnel.",
                "nullable": True
            }
        }
        self.output_type = "string"
        
        # Base de données (10 plats min)
        self.menu_data = [
            {"nom": "Salade César", "prix": 12, "prep": "10min", "allergenes": ["gluten", "lait"], "categorie": "entrée"},
            {"nom": "Soupe à l'oignon", "prix": 9, "prep": "15min", "allergenes": ["gluten"], "categorie": "entrée"},
            {"nom": "Carpaccio de Boeuf", "prix": 14, "prep": "10min", "allergenes": [], "categorie": "entrée"},
            {"nom": "Burger Classique", "prix": 18, "prep": "20min", "allergenes": ["gluten", "lait", "oeuf"], "categorie": "plat"},
            {"nom": "Risotto aux Champignons", "prix": 19, "prep": "25min", "allergenes": ["lait"], "categorie": "plat"},
            {"nom": "Pavé de Saumon", "prix": 22, "prep": "20min", "allergenes": ["poisson"], "categorie": "plat"},
            {"nom": "Curry de Légumes (Vegan)", "prix": 16, "prep": "20min", "allergenes": [], "categorie": "plat"},
            {"nom": "Steak Frites", "prix": 24, "prep": "15min", "allergenes": [], "categorie": "plat"},
            {"nom": "Fondant au Chocolat", "prix": 8, "prep": "15min", "allergenes": ["gluten", "lait", "oeuf"], "categorie": "dessert"},
            {"nom": "Salade de Fruits", "prix": 7, "prep": "10min", "allergenes": [], "categorie": "dessert"},
            {"nom": "Sorbet Citron", "prix": 6, "prep": "5min", "allergenes": [], "categorie": "dessert"},
        ]
        super().__init__()

    def forward(self, category: str = None, max_price: float = None, exclude_allergen: str = None) -> str:
        """Filtre le menu selon les critères."""
        results = []
        
        for item in self.menu_data:
            # Filtre Catégorie
            if category and category.lower() not in item["categorie"].lower():
                continue
            
            # Filtre Prix
            if max_price and item["prix"] > float(max_price):
                continue
                
            # Filtre Allergène / Régime
            if exclude_allergen:
                clean_exclude = exclude_allergen.lower()
                # Gestion spécifique pour "végétarien" -> on exclut viande/poisson implicitement si on veut, 
                # mais ici on simplifie en cherchant dans la liste des allergènes ou le nom
                item_allergens = [a.lower() for a in item["allergenes"]]
                
                # Si on demande "sans gluten" et que "gluten" est dans la liste
                if clean_exclude in item_allergens:
                    continue
                
                # Logique végétarienne simplifiée (si le plat est viande/poisson)
                if clean_exclude == "viande" or clean_exclude == "vegetarien":
                    if "boeuf" in item["nom"].lower() or "burger" in item["nom"].lower() or "steak" in item["nom"].lower() or "saumon" in item["nom"].lower():
                        continue

            results.append(f"- {item['nom']} ({item['categorie']}) : {item['prix']}€ | Allergènes: {', '.join(item['allergenes']) if item['allergenes'] else 'Aucun'}")

        if not results:
            return "Aucun plat ne correspond à vos critères."
        
        return "\n".join(results)

# Outil simple de calcul (via décorateur @tool)
@tool
def calculate(expression: str) -> str:
    """
    Évalue une expression mathématique simple pour calculer les totaux.
    Args:
        expression: L'expression mathématique (ex: '12 + 16 + 8').
    """
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Erreur: caractères non autorisés."
        return str(eval(expression))
    except Exception as e:
        return f"Erreur de calcul: {e}"

# =============================================================================
# 5.2 - AGENT AVEC PLANIFICATION
# =============================================================================

# Instanciation de l'outil base de données
menu_tool = MenuDatabaseTool()

# Création de l'agent
agent = CodeAgent(
    tools=[menu_tool, calculate],
    model=model,
    name="maitre_hotel_ia",
    description="Un serveur intelligent capable de créer des menus sur mesure.",
    planning_interval=2, # L'agent va re-planifier toutes les 2 étapes
)

# Système de logging manuel (Livrable .txt)
class FileLogger:
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write(f"--- TRACE D'EXECUTION {datetime.datetime.now()} ---\n\n")
    
    def log(self, role, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] [{role.upper()}]\n{content}\n{'-'*40}\n"
        print(entry) # Affichage console
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(entry)

logger = FileLogger("run_restaurant_trace.txt")

def run_agent_interaction(user_input, reset_memory=False):
    logger.log("user", user_input)
    
    # Si reset=False, l'agent garde l'historique (CodeAgent le gère nativement si on ne le recrée pas, 
    # mais ici on utilise .run(). Pour le conversationnel strict sans reset avec smolagents, 
    # on continue d'utiliser la même instance).
    if reset_memory:
        # CodeAgent n'a pas de méthode simple "clear_memory" publique exposée de manière standard 
        # sans toucher aux internes, donc on simule souvent en instanciant un nouvel agent 
        # ou en passant par des méthodes spécifiques.
        # Pour cet exercice, on assume que la méthode .run() nettoie si on ne gère pas l'état manuellement,
        # MAIS smolagents garde l'état en mode chat. 
        pass 

    response = agent.run(user_input)
    logger.log("agent", str(response))
    return response

# =============================================================================
# EXECUTION DES SCENARIOS
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("DEMARRAGE DU RESTAURANT INTELLIGENT")
    print("="*60)

    # SCENARIO 5.2 : Requête complexe avec contraintes
    print("\n--- 5.2 TEST DE PLANIFICATION ---")
    request_complex = (
        "On est 3. Un vegetarien, un sans gluten, et moi je mange de tout. "
        "Budget max 60 euros pour le groupe. Proposez-nous un menu complet (Entrée + Plat ou Plat + Dessert chacun)."
    )
    run_agent_interaction(request_complex)

    # SCENARIO 5.3 : Mode Conversationnel
    print("\n--- 5.3 DIALOGUE CONVERSATIONNEL ---")
    
    # Tour 1 : Demande de suggestions
    print("\n[Tour 1]")
    run_agent_interaction("J'aimerais juste un dessert pas cher.")

    # Tour 2 : Changement d'avis
    print("\n[Tour 2]")
    run_agent_interaction("Finalement, je préfère quelque chose au chocolat, peu importe le prix.")

    # Tour 3 : L'addition
    print("\n[Tour 3]")
    run_agent_interaction("Ok je prends ça. C'est combien ?")

    print(f"\nTrace sauvegardée dans 'run_restaurant_trace.txt'")