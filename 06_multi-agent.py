import os
import time
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, tool
from langfuse import observe, get_client
import litellm

# --- CONFIGURATION ---
GROUP_NAME = "GROUPE_NOA_NIELS"
load_dotenv()

litellm.callbacks = ["langfuse_otel"]

model = LiteLLMModel(
    model_id="groq/llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY")
)
langfuse = get_client()

# =============================================================================
# OUTILS ULTRA-OPTIMISÉS (Token Savers)
# =============================================================================

@tool
def check_dietary_info(dish: str) -> str:
    """
    Check allergens.
    
    Args:
        dish: Dish name.
    """
    # Version compressée
    return f"Analysis for {dish}: Check ingredients manually. Assume Soup/Salad/Risotto are safe if Vegetarian. Steak/Fish are NOT Vegetarian."

@tool
def check_fridge() -> str:
    """Returns ingredients."""
    return "Eggs, Milk, Butter, Veggies, Rice, Pasta, Salmon."

@tool
def get_menu_ideas(preference: str) -> str:
    """
    Get a FULL menu suggestion (Starter, Main, Dessert).
    
    Args:
        preference: Any preference (e.g. 'vegetarian', 'general').
    """
    # ASTUCE : On renvoie TOUT le menu d'un coup pour éviter 3 appels d'outils
    return """
    Suggested Menu:
    1. Appetizer: Cucumber Sticks (Veg, GF, Nut-free)
    2. Starter: Vegetable Soup (Veg, GF, Nut-free)
    3. Main: Mushroom Risotto (Veg, GF, Nut-free)
    4. Dessert: Fruit Salad (Veg, GF, Nut-free)
    """

@tool
def calculate_total_cost(menu_description: str) -> str:
    """
    Estimate total cost.
    
    Args:
        menu_description: Description of the full menu.
    """
    return "Total Estimated Cost: 95€ (fits within 120€ budget)."

# =============================================================================
# AGENTS (Strict & Rapides)
# =============================================================================

# On réduit max_steps à 2 pour qu'ils aillent droit au but
nutritionist = CodeAgent(
    tools=[check_dietary_info],
    model=model,
    name="nutritionist",
    description="Validates dietary constraints.",
    max_steps=2
)

chef_agent = CodeAgent(
    tools=[check_fridge, get_menu_ideas],
    model=model,
    name="chef_agent",
    description="Proposes the full menu.",
    max_steps=2
)

budget_agent = CodeAgent(
    tools=[calculate_total_cost],
    model=model,
    name="budget_agent",
    description="Validates budget.",
    max_steps=2
)

# =============================================================================
# MANAGER
# =============================================================================

def build_restaurant_manager():
    manager = CodeAgent(
        tools=[], 
        model=model,
        managed_agents=[chef_agent, nutritionist, budget_agent], 
        name="restaurant_manager",
        description="Manager.",
        max_steps=6 
    )
    return manager

# =============================================================================
# EXECUTION
# =============================================================================

@observe(name="run_restaurant_scenario")
def run_scenario():
    manager = build_restaurant_manager()
    
    # Prompt conçu pour limiter la verbosité
    system_instructions = """
    RULES:
    1. Delegate to 'chef_agent' to get a FULL MENU in ONE shot.
    2. Delegate to 'budget_agent' to check cost.
    3. Call agents with ONE string argument only.
    4. Be extremely concise.
    """

    user_request = """
    TASK: Plan a meal for 8 people (2 veg, 1 gluten-free). Budget 120€.
    """
    
    full_task = system_instructions + "\n" + user_request
    
    print(f"REQUÊTE ENVOYÉE :\n{user_request}\n")
    print("Le Manager démarre (Version Optimisée)...\n")
    
    result = manager.run(full_task)
    return result

if __name__ == "__main__":
    print("=" * 60)
    print(f"MULTI-AGENT RESTAURANT SYSTEM - {GROUP_NAME}")
    print("=" * 60)

    try:
        final_response = run_scenario()
        print("\n" + "="*60)
        print("RÉPONSE FINALE :")
        print(final_response)
        print("="*60)
    except Exception as e:
        print(f"\nERREUR : {e}")
        print("Si c'est une erreur 429 ou RateLimit, attendez 60 secondes.")

    langfuse.flush()