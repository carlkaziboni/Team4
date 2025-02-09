import openai
import base64

client = openai.OpenAI(api_key= "")

def analyse(image_path):
    user_prompt = """
    Diageo is a British multinational alcoholic beverage company.

    1. Identify opportunities for renewable energy integration in Scope 1 & 2 operations.
    2. Propose solutions for efficient energy sourcing, usage, and monitoring.
    3. Ensure scalability, cost-effectiveness, and reliability in transitioning to 100% renewable energy.
    IMPORTANT: Response must be under 50 words, in 3 bullet points, formal, and direct.
    """
    
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content