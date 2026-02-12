import json
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from langfuse import observe, get_client, Evaluation
GROUP_NAME="GROUPE_NOA_NIELS"
load_dotenv()
groq_client = Groq()
langfuse = get_client()

# =============================================================================
# 1. LE CHEFBOT (La Task à évaluer)
# =============================================================================
@observe()
def chef_bot_task(constraints: str) -> str:
    # Ici, ton appel LLM pour générer la recette
    return "Menu suggéré : Salade de pois chiches, feta et concombres. Sauce citronnée. Sans arachides et conforme au budget."

# =============================================================================
# 2. EVALUATEUR 1 : PROGRAMMATIQUE (Règles strictes)
# =============================================================================
def rule_evaluator(**kwargs) -> list:
    """Vérifie strictement les ingrédients interdits et requis (Code Python)"""
    output = kwargs.get("output")
    expected = kwargs.get("expected_output")
    
    output_lower = output.lower()
    
    # Check Must Avoid
    must_avoid = expected.get("must_avoid", [])
    forbidden_found = [i for i in must_avoid if i.lower() in output_lower]
    evaluator_safety_score = 1.0 if not forbidden_found else 0.0

    # Check Must Include
    must_include = expected.get("must_include", [])
    included_found = [i for i in must_include if i.lower() in output_lower]
    evaluator_inclusion_score = len(included_found) / len(must_include) if must_include else 1.0

    return [
        Evaluation(name="evaluator_safety_score", value=evaluator_safety_score, comment=f"Interdits trouvés: {forbidden_found}"),
        Evaluation(name="evaluator_inclusion_score", value=evaluator_inclusion_score)
    ]

# =============================================================================
# 3. EVALUATEUR 2 : LLM JUDGE (Subjectif)
# =============================================================================
@observe(name="llm-judge", as_type="generation")
def llm_judge_task(input_text: str, output: str, expected: dict) -> dict:
    prompt = f"""Tu es un Expert culinaire, note de 0.0 à 1.0 :
    Contraintes: {input_text}
    Réponse: {output}
    Cibles: {expected}
    Réponds en JSON uniquement : {{"llm_pertinence": 0.0, "llm_creativite": 0.0, "llm_praticite": 0.0}}"""
    
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def llm_evaluator(**kwargs) -> list:
    """Wrapper pour convertir le résultat du juge LLM en objets Evaluation"""
    output = kwargs.get("output")
    expected = kwargs.get("expected_output")
    input_data = kwargs.get("input")
    
    scores = llm_judge_task(input_data["constraints"], output, expected)
    
    return [
        Evaluation(name="llm_pertinence", value=scores["llm_pertinence"]),
        Evaluation(name="llm_creativite", value=scores["llm_creativite"]),
        Evaluation(name="llm_praticite", value=scores["llm_praticite"])
    ]

# =============================================================================
# 4. LANCEMENT DE L'EXPÉRIENCE (Regroupement)
# =============================================================================

@observe(name="chefbot_full_experiment")
def run_full_experiment():
    langfuse.update_current_trace(
            name=f"{GROUP_NAME}, Partie 3",
            tags=[GROUP_NAME, "Partie 3"],
            metadata={
                "type":"evaluation",
            }
        )
    dataset = langfuse.get_dataset("chefbot-menu-eval-niels-noa")

    langfuse.run_experiment(
        name=f"chefbot-full-eval-{datetime.now().strftime('%H%M%S')}",
        data=dataset.items,
        task=lambda item: chef_bot_task(item.input["constraints"]),
        evaluators=[rule_evaluator, llm_evaluator], # Les deux évaluateurs tournent en parallèle
        description="Comparaison Rules vs LLM Judge"
    )

run_full_experiment()
langfuse.flush()
