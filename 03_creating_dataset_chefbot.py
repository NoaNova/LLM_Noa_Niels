from dotenv import load_dotenv
from langfuse import observe, get_client, Evaluation
from groq import Groq
import json
from datetime import datetime
from typing import Callable

load_dotenv()

groq_client = Groq()

GROUP_NAME="GROUPE_NOA_NIELS"
# =============================================================================
# CREATING DATASETS
# =============================================================================
@observe(name="create_chefbot_menu_eval_dataset-niels-noa")
def create_sentiment_dataset():
    get_client().update_current_trace(
            name=f"{GROUP_NAME}, Partie 3",
            tags=[GROUP_NAME, "Partie 3"],
            metadata={
                "type":"create_dataset",
            }
        )

    dataset = get_client().create_dataset(
        name="chefbot-menu-eval-niels-noa",
        description="Benchmark dataset for chefbot menu ",
        metadata={
            "created_by": "lecture_demo",
            "domain": "product_reviews",
            "version": "1.0"
        }
    )

    test_cases = [
    {
        "input": {"constraints": "Repas pour diabétique de type 2, budget modéré, sans cuisson"},
        "expected_output": {
            "must_avoid": ["sucre", "pates blanches", "pain blanc", "miel"],
            "must_include": ["légumes croquants", "protéines maigres", "noix"],
            "target_calories": 550,
            "price_per_meal": 8.0
        },
        "metadata": {
            "category": "health",
            "diet_type": "low-glycemic",
            "difficulty": "medium",
        }
    },
    {
        "input": {"constraints": "Allergie sévère aux arachides et sésame, budget étudiant, 4 personnes"},
        "expected_output": {
            "must_avoid": ["cacahuètes", "huile d'arachide", "graines de sésame", "tahin"],
            "must_include": ["pâtes complètes", "légumineuses", "œufs"],
            "target_calories": 750,
            "price_per_meal": 3.5
        },
        "metadata": {
            "category": "allergy",
            "diet_type": "standard",
            "difficulty": "high",
        }
    },
    {
        "input": {"constraints": "Vegan, sans gluten, focus fer et vitamine C pour absorption"},
        "expected_output": {
            "must_avoid": ["viande", "produits laitiers", "œufs", "blé", "orge"],
            "must_include": ["lentilles", "citron", "poivrons", "tofu"],
            "target_calories": 600,
            "price_per_meal": 12.0
        },
        "metadata": {
            "category": "lifestyle",
            "diet_type": "vegan-gf",
            "difficulty": "medium",
        }
    },
    {
        "input": {"constraints": "Menu de fête japonais (Omakase), budget luxe, pas de poisson cru"},
        "expected_output": {
            "must_avoid": ["sashimi", "sushi cru", "tartare de thon"],
            "must_include": ["boeuf wagyu", "tempura de crevettes", "soupe miso", "riz vinaigré"],
            "target_calories": 950,
            "price_per_meal": 60.0
        },
        "metadata": {
            "category": "cultural",
            "diet_type": "japanese",
            "difficulty": "easy",
        }
    },
    {
        "input": {"constraints": "Prise de masse (Bulking), riche en protéines, pas de produits laitiers, rapide (15min)"},
        "expected_output": {
            "must_avoid": ["fromage", "crème", "beurre", "lait de vache"],
            "must_include": ["poulet", "riz blanc", "avocat", "beurre d'amande"],
            "target_calories": 1100,
            "price_per_meal": 10.0
        },
        "metadata": {
            "category": "fitness",
            "diet_type": "high-protein",
            "difficulty": "medium",
        }
    }
    ]

    for i, case in enumerate(test_cases):
        get_client().create_dataset_item(
            dataset_name="chefbot-menu-eval-niels-noa",
            input=case["input"],
            expected_output=case["expected_output"],
            metadata=case["metadata"]
        
        )

    print(f"✓ Created dataset with {len(test_cases)} test cases")
    return dataset
create_sentiment_dataset()
