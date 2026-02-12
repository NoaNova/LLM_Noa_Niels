import os
import json
from dotenv import load_dotenv
from groq import Groq
from langfuse import observe, get_client

GROUP_NAME = "GROUPE_NOA_NIELS"

load_dotenv()
groq_client = Groq()
langfuse = get_client()

@observe(name="plan_weekly_menu") 
def plan_weekly_menu(constraints: str) -> dict:
    
    langfuse.update_current_trace(
        name=f"{GROUP_NAME}, Partie 2 - Menu",
        tags=[GROUP_NAME, "Partie 2", "Chef Planner"],
        metadata={
            "constraints": constraints
        }
    )

    try:
        # --- Étape 1 : Planification (avec Retry) ---
        print("1. Planification en cours...")
        plan_data = _plan_steps(constraints)
        
        steps = plan_data.get("steps", [])

        # --- Étape 2 : Exécution ---
        step_results = []
        for i, step in enumerate(steps):
            print(f"2.{i+1} Exécution : {step}")
            result = _execute_step(step, i, context=step_results)
            step_results.append(result)

        # --- Étape 3 : Synthèse ---
        print("3. Synthèse du menu...")
        final_menu = _synthesize_menu(constraints, step_results)

        return {
            "constraints": constraints,
            "plan": steps,
            "step_results": step_results,
            "final_answer": final_menu,
            "status": "success"
        }

    except Exception as e:
        # Gestion d'erreur robuste au niveau parent
        langfuse.update_current_span(
            level="ERROR",
            status_message=f"Erreur critique dans le planificateur : {str(e)}"
        )
        return {"status": "error", "error": str(e)}


@observe(name="planning", as_type="generation")
def _plan_steps(constraints: str) -> dict:
    """Génère le plan JSON avec 1 retry en cas d'erreur."""
    
    max_retries = 2 # 1 essai initial + 1 retry
    
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a methodical Chef. Break down the task of creating a weekly menu into 3 to 4 logical steps.
                        RETURN ONLY JSON format: {"steps": ["step description 1", "step description 2", ...]}
                        Do not add markdown formatting."""
                    },
                    {"role": "user", "content": f"Constraints for the menu: {constraints}"}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Tentative de parsing
            return json.loads(response.choices[0].message.content)

        except json.JSONDecodeError as e:
            # Log l'erreur dans Langfuse sans casser le programme immédiatement
            langfuse.update_current_span(
                level="ERROR",
                status_message=f"JSON invalide (Tentative {attempt+1}/{max_retries}): {str(e)}"
            )
            print(f"Erreur JSON (Essai {attempt+1}), nouvelle tentative...")
            
            # Si c'était la dernière tentative, on remonte l'erreur
            if attempt == max_retries - 1:
                raise e


@observe(name="execute_step", as_type="generation")
def _execute_step(step: str, index: int, context: list) -> dict:
    context_str = "\n".join([f"- Résultat précédent ({r['step']}): {r['output']}" for r in context])
    
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are a Chef executing a specific part of menu planning. Be concise and practical."
            },
            {
                "role": "user", 
                "content": f"Previous work:\n{context_str}\n\nCURRENT TASK: {step}\nExecute this task now."
            }
        ],
        temperature=0.5
    )

    return {
        "step_index": index,
        "step": step,
        "output": response.choices[0].message.content
    }


@observe(name="synthesis", as_type="generation")
def _synthesize_menu(constraints: str, results: list) -> str:
    full_context = "\n".join([f"Step {r['step']}: {r['output']}" for r in results])

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are a Chef. Based on all the planning steps, generate the final clear Weekly Menu presentation."
            },
            {
                "role": "user", 
                "content": f"Customer Constraints: {constraints}\n\nPlanning Data:\n{full_context}\n\nPlease output the Final Weekly Menu nicely formatted."
            }
        ],
        temperature=0.5
    )
    
    return response.choices[0].message.content


# --- Exécution ---
print("\n--- Lancement du Chef Planificateur (Mode Robuste) ---")
resultat = plan_weekly_menu("Menu végétarien pour 2 personnes, budget serré, incluant des restes pour le midi")

if resultat['status'] == 'success':
    print("PLAN GÉNÉRÉ :")
    print(resultat['plan'])
    print("MENU FINAL :")
    print(resultat['final_answer'])
else:
    print(f"Erreur Fatale : {resultat.get('error')}")

langfuse.flush()