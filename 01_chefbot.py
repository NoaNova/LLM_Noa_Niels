from dotenv import load_dotenv
from groq import Groq
from langfuse import observe, get_client

GROUP_NAME = "GROUPE_NOA_NIELS"

load_dotenv()
langfuse = get_client()

groq_client = Groq()


@observe(name="ask_chef", as_type="generation")
def ask_chef(question: str) -> str:
    # On définit la liste ici
    temperatures = [0.1, 0.7, 1.2]
    resultat_complet = ""

    # On doit boucler car groq.create n'accepte qu'un seul nombre (float) à la fois
    for temp in temperatures:
        langfuse.update_current_trace(
            name=f"{GROUP_NAME}, Partie 1",
            tags=[GROUP_NAME, "Partie 1"],
            metadata={
                "type":"ask_chef",
                "temperature": temp,
            }
        )

        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a specialized french chef with a deep knowledge of seasonal ingredients and traditional recipes. You provide concise and practical cooking advice based on the user's question."
                },
                {"role": "user", "content": question}
            ],
            temperature=temp  # On utilise la variable temp (un chiffre) de la boucle
        )
        
        # On ajoute la réponse au résultat final
        resultat_complet += f"\n--- Temperature {temp} ---\n{response.choices[0].message.content}\n"

    return resultat_complet

# Un seul appel ici, comme demandé
print(ask_chef("What are some quick and easy meals I can make for dinner?"))
langfuse.flush()