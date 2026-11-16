import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def main():
    information = """
    Elon Reeve Musk (/ˈiːlɒn/ ⓘ EE-lon; born June 28, 1971) is an industrialist, futurist and former United States public official who is the chief executive officer and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and became the only trillionaire (in terms of US dollars) in June 2026; as of November 2026, Forbes estimates his net worth to be $849.7 billion.

    Musk was born into the wealthy Musk family in Pretoria, South Africa. He emigrated to Canada in 1989; he has Canadian citizenship as his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California, United States, to pursue business ventures. In 1995, he co-founded Zip2, a web software company. Following its sale in 1999, he co-founded X.com, an e-commerce payment system that merged with Confinity in March 2000 to form PayPal. Musk also became an American citizen in 2002.

    In 2002, Musk founded and became CEO and chief engineer of SpaceX, a space technology company. Musk joined Tesla as an angel investor in 2004 and became its CEO and product architect in 2008. In 2015, Musk co-founded OpenAI to advance artificial intelligence (AI) research, but later left; his growing discontent with the organization's direction and leadership in the AI boom in the 2020s led him to establish xAI, which became a subsidiary of SpaceX in 2026. In 2022, he acquired Twitter, a social networking service, and made significant changes, including rebranding it as X in 2023. His other businesses include Neuralink, a neurotechnology company that he co-founded in 2016, and the Boring Company, a tunneling company that he founded in 2017. In November 2025, Tesla approved a pay package worth $1 trillion for Musk, which he is to receive over 10 years if certain milestones are met, such as achieving a market capitalization of $8.5 trillion. Musk became the first US-dollar trillionaire upon the initial public offering of SpaceX in June 2026, though a SpaceX stock collapse brought him under the threshold by the next month.

    Musk is a supporter of global far-right politics, figures, and political parties. His political activities, statements and views have made him a polarizing figure. He has been criticized for making unscientific, false, and misleading statements, promoting conspiracy theories, and affirming antisemitic, white nationalist, and transphobic comments. His acquisition of Twitter was controversial because, following his pledge to decrease censorship, there was an increase in hate speech and misinformation on the service. His role in the second Trump administration attracted public backlash, particularly in response to DOGE and its cuts to the US Agency for International Development (USAID).

    Musk was the largest donor in the 2024 United States presidential election, in which he supported Republican nominee Donald Trump. After Trump's inauguration in January 2025, Musk was a Senior Advisor to President Trump and the de facto head of the Department of Government Efficiency (DOGE). Musk left the Trump administration in May 2025 and returned to managing his companies; shortly thereafter he had a public feud with Trump. Musk also became the largest political donor in American history in 2026, when he supported Republican nominees in the midterm elections.

    Early life and education
    See also: Musk family
    Elon Reeve Musk was born on June 28, 1971, in Pretoria, Transvaal Province (Gauteng Province since 1994) South Africa's administrative capital.[3][4] He is of British and Pennsylvania Dutch ancestry.[5][6] His mother, Maye (née Haldeman), is a model and dietitian born in Saskatchewan, Canada, and raised in South Africa.[7][8][9] Musk therefore holds both South African and Canadian citizenship from birth.[10] His father, Errol Musk, is a South African electromechanical engineer, pilot, sailor, consultant, emerald dealer, and property developer, who partly owned a rental lodge at Timbavati Private Nature Reserve.[11][12][13][14]

    His maternal grandfather, Joshua N. Haldeman, who died in a plane crash when Musk was a toddler, was an American-born Canadian chiropractor, aviator and political activist in the technocracy movement[15][16] who moved to South Africa in 1950.[17] Musk has a younger brother, Kimbal, a younger sister, Tosca, and four paternal half-siblings.[18][19][9][20] He was baptized as a child in the Anglican Church of Southern Africa.[21][22] The Musk family was wealthy during his youth.[14] Despite both Musk and Errol previously stating that Errol was a part owner of a Zambian emerald mine,[14] in 2023, Errol recounted that the deal he made was to receive "a portion of the emeralds produced at three small mines".[23][24] Errol was elected to the Pretoria City Council as a representative of the anti-apartheid Progressive Party and has said that his children shared their father's dislike of apartheid.[3]

    After his parents divorced in 1979, Musk, aged around 9, chose to live with his father because he had an Encyclopædia Britannica set and a computer.[25][5][11] He later regretted his decision and became estranged from his father.[26] Musk has recounted trips to a wilderness school that he described as a "paramilitary Lord of the Flies" where "bullying was a virtue" and children were encouraged to fight over rations.[27] In one incident, after an altercation with a fellow pupil, Musk was thrown down concrete steps and beaten severely, leading to him being hospitalized for his injuries.[28] He described his father berating him after he was discharged from the hospital.[28] Errol denied berating Musk.[29]

    Musk was an enthusiastic reader of books, and had attributed his success in part to having read The Lord of the Rings, the Foundation series, and The Hitchhiker's Guide to the Galaxy.[13][30] At age ten, he developed an interest in computing and video games, teaching himself how to program from the VIC-20 user manual.[31] At age twelve, in 1983, Musk sold his BASIC-based game Blastar to PC and Office Technology magazine for approximately $500 (equivalent to $1,600 in 2025).[32][33]
    """
    
    summary_template = """
        Given the information {information} about a person I want you to create:
        1. A short summary of the person in 2-3 sentences.
        2. A list of 5 interesting facts about the person.
    """
    
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )
    
    llm = ChatGoogleGenerativeAI(
        temperature=0,
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_AI_API_KEY"),
    )
    chain = summary_prompt_template | llm
    result = chain.invoke({"information": information})
    print(result.content)
    


if __name__ == "__main__":
    main()