import json
import os
import re
from dotenv import load_dotenv
from groq import Groq
from langfuse import observe, get_client
from smolagents import CodeAgent, LiteLLMModel, tool

# --- CONFIGURATION ---
GROUP_NAME = "GROUPE_NOA_NIELS"
load_dotenv()

# 1. Client pour la boucle manuelle (Partie 4.2)
groq_client = Groq()

# 2. Modèle pour Smolagents (Partie 4.3)
# On utilise LiteLLM pour connecter Smolagents à Groq
model = LiteLLMModel(
    model_id="groq/llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY")
)
langfuse = get_client()

# =============================================================================
# DEFINITION DES OUTILS "CHEF" (Compatible Smolagents & Manuel)
# =============================================================================

@tool
def get_seasonal_products(month: str) -> str:
    """
    Returns the list of seasonal ingredients for a specific month in France.
    
    Args:
        month: The month name (e.g., 'March', 'June').
    """
    data = {
        "March": "Asparagus, Spinach, Radish, Lemon, Kiwi",
        "June": "Strawberry, Zucchini, Tomato, Cherry, Apricot",
        "October": "Pumpkin, Mushroom, Apple, Pear, Grapes",
        "December": "Chestnut, Truffle, Scallop, Clementine",
    }
    return data.get(month, f"No data available for {month}")

@tool
def calculate_food_cost(price_per_kg: float, weight_kg: float) -> str:
    """
    Calculates the total cost of an ingredient.
    
    Args:
        price_per_kg: The price of the ingredient per kilogram.
        weight_kg: The weight of the ingredient in kilograms.
    """
    try:
        total = float(price_per_kg) * float(weight_kg)
        return str(round(total, 2))
    except Exception as e:
        return f"Error in calculation: {e}"

@tool
def get_reservations(date: str) -> str:
    """
    Get restaurant table reservations for a specific date.
    
    Args:
        date: The date to check in format 'DD/MM/YYYY'.
    """
    # Outil "Fragile" qui impose un format strict
    if not re.match(r"^\d{2}/\d{2}/\d{4}$", date):
        return f"ERROR: invalid date format '{date}'. Expected DD/MM/YYYY."

    fake_bookings = {
        "15/03/2025": "FULL SERVICE: 45 covers (VIP table booked)",
        "16/03/2025": "Light service: 12 covers",
    }
    return fake_bookings.get(date, f"No reservations found for {date}")

# =============================================================================
# PARTIE 4.2 : BOUCLE MANUELLE (Sans Framework)
# =============================================================================

# Définition JSON manuelle pour l'API Groq (car Groq ne lit pas @tool directement)
manual_tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_seasonal_products",
            "description": "Get seasonal ingredients for a month.",
            "parameters": {"type": "object", "properties": {"month": {"type": "string"}}, "required": ["month"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_food_cost",
            "description": "Calculate total ingredient cost.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "price_per_kg": {"type": "number"},
                    "weight_kg": {"type": "number"}
                }, 
                "required": ["price_per_kg", "weight_kg"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_reservations",
            "description": "Get reservations for a date.",
            "parameters": {"type": "object", "properties": {"date": {"type": "string", "description": "DD/MM/YYYY"}}, "required": ["date"]}
        }
    }
]

# Registre manuel pour mappper les noms JSON vers les fonctions Python
TOOL_REGISTRY = {
    "get_seasonal_products": get_seasonal_products,
    "calculate_food_cost": calculate_food_cost,
    "get_reservations": get_reservations,
}

@observe(name="manual_loop")
def run_manual_loop(question: str):
    print(f"[MANUAL CHEF] Question: {question}")
    
    langfuse.update_current_trace(
        name=f"{GROUP_NAME}, Partie 4.2 - Manual Loop",
        tags=[GROUP_NAME, "Partie 4.2", "Manual"],
    )

    messages = [
    {
        "role": "system", 
        "content": "You are a Head Chef. Use tools to manage the kitchen. IMPORTANT: Once you have the tool results, you MUST summarize the information in your final response to the user."
    },
    {"role": "user", "content": question}
]
    for i in range(5):
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=manual_tools_schema,
            tool_choice="auto"
        )
        msg = response.choices[0].message
        
        # Si pas d'appel d'outil, c'est fini
        if not msg.tool_calls:
            print(f"[MANUAL] Réponse finale : {msg.content}")
            return msg.content
            
        # 1. Convertir en dictionnaire
        msg_dict = msg.model_dump()
        
        # 2. Liste noire des champs que l'API Groq refuse
        # On supprime tout ce qui n'est pas standard
        keys_to_remove = ["annotations", "executed_tools", "function_call", "tool_calls_info"]
        
        for key in keys_to_remove:
            if key in msg_dict:
                del msg_dict[key]

        messages.append(msg_dict)
        # ----------------------

        # Exécuter les outils
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"[MANUAL] Tool Call: {fn_name}({args})")
            
            func = TOOL_REGISTRY.get(fn_name)
            if func:
                result = str(func(**args)) 
            else:
                result = f"Error: Tool {fn_name} not found"
                
            print(f"   -> Result: {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
            
    return "Error: Max iterations reached"


# =============================================================================
# PARTIE 4.3 : MIGRATION SMOLAGENTS
# =============================================================================

@observe(name="smolagents_loop")
def run_smolagents_loop(question: str):
    print(f"[SMOLAGENTS CHEF] Question: {question}")
    
    langfuse.update_current_trace(
        name=f"{GROUP_NAME}, Partie 4.3 - Smolagents",
        tags=[GROUP_NAME, "Partie 4.3", "SmolAgents"],
    )

    # Création de l'agent
    agent = CodeAgent(
        tools=[get_seasonal_products, calculate_food_cost, get_reservations],
        model=model,
        max_steps=5,
        add_base_tools=False
    )

    result = agent.run(question)
    
    print(f"[SMOLAGENTS] Réponse finale : {result}")
    return result


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Scénario Chef : On veut les ingrédients de Mars ET vérifier les résas pour le 15 Mars
    task = "What represent the seasonal products for March? Also, check the reservations for 15/03/2025."
    
    # 1. Execution Manuelle
    run_manual_loop(task)

    print("\n" + "="*50 + "\n")

    # 2. Execution Smolagents
    run_smolagents_loop(task)

    print("\n" + "="*50)
    print("OBSERVATIONS :")
    print("- La boucle manuelle nécessite de définir les schémas JSON et gérer l'historique des messages 'tool'.")
    print("- Smolagents génère automatiquement le code Python pour appeler les outils, simplifiant grandement l'implémentation.")
    
    langfuse.flush()